#include <array>
#include <algorithm>
#include <atomic>
#include <chrono>
#include <cctype>
#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <future>
#include <iostream>
#include <memory>
#include <optional>
#include <set>
#include <sstream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <thread>
#include <unordered_map>
#include <utility>
#include <vector>

#include "cli/diagnostics_parity.hpp"
#include "compiler/parser.hpp"
#include "compiler/lower_to_runtime.hpp"
#include "runtime/json.hpp"
#include "runtime/compiled_rules.hpp"
#include "puzzlescript/compiler.h"
#include "puzzlescript/puzzlescript.h"

#ifdef PS_HAVE_SDL2
int puzzlescript_cpp_run_player_for_ir(const std::string& irPath);
int puzzlescript_cpp_run_player_for_game(ps_game* game, const std::string& saveKey);
#endif

namespace {

bool sessionCreateForGame(ps_game* game, const std::optional<std::string>& loadedLevelSeed, ps_full_state** outSession, ps_error** outError) {
    if (loadedLevelSeed.has_value()) {
        return ps_full_state_create_with_loaded_level_seed(game, loadedLevelSeed->c_str(), outSession, outError);
    }
    return ps_full_state_create(game, outSession, outError);
}

std::optional<std::string> findArgValue(const std::vector<std::string>& args, const char* flag) {
    for (size_t index = 0; index + 1 < args.size(); ++index) {
        if (args[index] == flag) {
            return args[index + 1];
        }
    }
    return std::nullopt;
}

#ifndef PS_NODE_EXECUTABLE
#define PS_NODE_EXECUTABLE "node"
#endif

#ifndef PS_EXPORT_IR_SCRIPT
#define PS_EXPORT_IR_SCRIPT "src/tests/js_oracle/export_ir_json.js"
#endif

#ifndef PS_EXPORT_TRACE_SCRIPT
#define PS_EXPORT_TRACE_SCRIPT "src/tests/js_oracle/export_execution_trace.js"
#endif

std::string readFile(const std::filesystem::path& path) {
    std::ifstream stream(path, std::ios::binary);
    if (!stream) {
        throw std::runtime_error("Failed to open file: " + path.string());
    }
    std::ostringstream buffer;
    buffer << stream.rdbuf();
    return buffer.str();
}

bool writeFileIfChanged(const std::filesystem::path& path, const std::string& text) {
    if (std::filesystem::exists(path)) {
        if (readFile(path) == text) {
            return false;
        }
    }
    if (!path.parent_path().empty()) {
        std::filesystem::create_directories(path.parent_path());
    }
    std::ofstream stream(path, std::ios::binary);
    if (!stream) {
        throw std::runtime_error("Failed to write file: " + path.string());
    }
    stream << text;
    return true;
}

std::string stripTrailingJsonCommas(std::string_view input) {
    std::string out;
    out.reserve(input.size());
    bool inString = false;
    bool escaped = false;
    for (size_t index = 0; index < input.size(); ++index) {
        const char ch = input[index];
        if (inString) {
            out.push_back(ch);
            if (escaped) {
                escaped = false;
            } else if (ch == '\\') {
                escaped = true;
            } else if (ch == '"') {
                inString = false;
            }
            continue;
        }
        if (ch == '"') {
            inString = true;
            out.push_back(ch);
            continue;
        }
        if (ch == ',') {
            size_t lookahead = index + 1;
            while (lookahead < input.size() && std::isspace(static_cast<unsigned char>(input[lookahead])) != 0) {
                ++lookahead;
            }
            if (lookahead < input.size() && (input[lookahead] == ']' || input[lookahead] == '}')) {
                continue;
            }
        }
        out.push_back(ch);
    }
    return out;
}

puzzlescript::json::Value loadJsDataArrayAsJson(const std::filesystem::path& path) {
    const std::string text = readFile(path);
    const size_t begin = text.find('[');
    const size_t end = text.rfind(']');
    if (begin == std::string::npos || end == std::string::npos || end < begin) {
        throw std::runtime_error("Could not find JSON array in: " + path.string());
    }
    const std::string jsonish = stripTrailingJsonCommas(std::string_view(text).substr(begin, end - begin + 1));
    puzzlescript::json::Value root = puzzlescript::json::parse(jsonish);
    if (!root.isArray()) {
        throw std::runtime_error("Expected top-level test data array in: " + path.string());
    }
    return root;
}

const puzzlescript::json::Value& requireField(const puzzlescript::json::Value::Object& object, std::string_view key) {
    const auto it = object.find(std::string(key));
    if (it == object.end()) {
        throw std::runtime_error("Missing required field: " + std::string(key));
    }
    return it->second;
}

int32_t looseAsInt(const puzzlescript::json::Value& value) {
    if (value.isInteger()) {
        return static_cast<int32_t>(value.asInteger());
    }
    if (value.isDouble()) {
        return static_cast<int32_t>(value.asDouble());
    }
    if (value.isString()) {
        return static_cast<int32_t>(std::stoi(value.asString()));
    }
    return 0;
}

std::vector<int32_t> looseAsIntArray(const puzzlescript::json::Value& value) {
    std::vector<int32_t> result;
    if (!value.isArray()) {
        return result;
    }
    const auto& array = value.asArray();
    result.reserve(array.size());
    for (const auto& item : array) {
        if (item.isInteger() || item.isDouble() || item.isString()) {
            result.push_back(looseAsInt(item));
        }
    }
    return result;
}

void appendIntList(std::ostream& stream, const std::vector<int32_t>& values) {
    for (size_t index = 0; index < values.size(); ++index) {
        if (index > 0) {
            stream << ",";
        }
        stream << values[index];
    }
}

std::optional<int32_t> optionalIntField(const puzzlescript::json::Value::Object& object, std::string_view key) {
    const auto it = object.find(std::string(key));
    if (it == object.end() || it->second.isNull()) {
        return std::nullopt;
    }
    return looseAsInt(it->second);
}

std::optional<std::string> optionalStringField(const puzzlescript::json::Value::Object& object, std::string_view key) {
    const auto it = object.find(std::string(key));
    if (it == object.end() || it->second.isNull()) {
        return std::nullopt;
    }
    return it->second.asString();
}

struct ScopedEnvSilence {
    explicit ScopedEnvSilence(std::vector<std::string> names)
        : names(std::move(names)) {
        values.reserve(this->names.size());
        for (const auto& name : this->names) {
            const char* value = std::getenv(name.c_str());
            if (value) {
                values.emplace_back(value);
                unsetenv(name.c_str());
            } else {
                values.emplace_back(std::nullopt);
            }
        }
    }

    ~ScopedEnvSilence() {
        for (size_t index = 0; index < names.size(); ++index) {
            const auto& name = names[index];
            const auto& value = values[index];
            if (value.has_value()) {
                setenv(name.c_str(), value->c_str(), 1);
            } else {
                unsetenv(name.c_str());
            }
        }
    }

    std::vector<std::string> names;
    std::vector<std::optional<std::string>> values;
};

std::vector<std::string> loadInputTokensFromJsonText(const std::string& jsonText) {
    std::vector<std::string> tokens;
    const auto root = puzzlescript::json::parse(jsonText);
    if (!root.isArray()) {
        throw std::runtime_error("inputs-json must be a JSON array");
    }
    const auto& array = root.asArray();
    tokens.reserve(array.size());
    for (const auto& value : array) {
        if (value.isString()) {
            tokens.push_back(value.asString());
        } else if (value.isInteger() || value.isDouble()) {
            tokens.push_back(std::to_string(looseAsInt(value)));
        } else {
            tokens.push_back("0");
        }
    }
    return tokens;
}

std::vector<std::string> inputTokensFromJsonArray(const puzzlescript::json::Value& root) {
    if (!root.isArray()) {
        throw std::runtime_error("input trace must be a JSON array");
    }
    const auto& array = root.asArray();
    std::vector<std::string> tokens;
    tokens.reserve(array.size());
    for (const auto& value : array) {
        if (value.isString()) {
            tokens.push_back(value.asString());
        } else if (value.isInteger() || value.isDouble()) {
            tokens.push_back(std::to_string(looseAsInt(value)));
        } else {
            tokens.push_back("0");
        }
    }
    return tokens;
}

std::vector<std::string> loadInputTokensFromJsonFile(const std::filesystem::path& path) {
    return loadInputTokensFromJsonText(readFile(path));
}

std::optional<ps_input> parseInputToken(const std::string& token);

struct ReplayOracleStats {
    uint64_t compactTurnChecks = 0;
    uint64_t compactTurnHandled = 0;
    uint64_t compactTurnStateChecks = 0;
    uint64_t compactTurnFailures = 0;
    std::string firstFailure;
};

std::string compactTurnOracleMismatchSummary(const ps_compact_turn_oracle_info& info) {
    std::ostringstream out;
    out << " handled=" << (info.handled ? 1 : 0)
        << " state_checked=" << (info.state_checked ? 1 : 0)
        << " compact(changed=" << (info.compact_result.changed ? 1 : 0)
        << ",won=" << (info.compact_result.won ? 1 : 0)
        << ",restarted=" << (info.compact_result.restarted ? 1 : 0)
        << ",transitioned=" << (info.compact_result.transitioned ? 1 : 0)
        << ") interpreter(changed=" << (info.interpreter_result.changed ? 1 : 0)
        << ",won=" << (info.interpreter_result.won ? 1 : 0)
        << ",restarted=" << (info.interpreter_result.restarted ? 1 : 0)
        << ",transitioned=" << (info.interpreter_result.transitioned ? 1 : 0)
        << ")";
    return out.str();
}

bool replayInputTokens(
    ps_full_state* session,
    const std::vector<std::string>& tokens,
    std::vector<std::string>* outSounds,
    ReplayOracleStats* oracleStats = nullptr
) {
    if (!session) {
        return false;
    }
    if (outSounds) {
        outSounds->clear();
    }
    for (const auto& token : tokens) {
        if (token == "undo") {
            (void)ps_full_state_undo(session);
            continue;
        }
        if (token == "restart") {
            (void)ps_full_state_restart(session);
            for (int againPass = 0; againPass < 500 && ps_full_state_pending_again(session); ++againPass) {
                const ps_step_result stepResult = ps_full_state_turn(session, PS_INPUT_TICK);
                if (outSounds && stepResult.audio_event_count > 0 && stepResult.audio_events) {
                    for (size_t i = 0; i < stepResult.audio_event_count; ++i) {
                        const ps_audio_event& event = stepResult.audio_events[i];
                        if (event.kind && event.kind[0] != '\0') {
                            outSounds->push_back(event.kind);
                        }
                    }
                }
            }
            continue;
        }
        const auto input = parseInputToken(token);
        if (!input.has_value()) {
            std::cerr << "Unsupported replay input token: " << token << "\n";
            return false;
        }

        if (oracleStats != nullptr) {
            ps_compact_turn_oracle_info oracleInfo{};
            if (!ps_full_state_compact_turn_oracle_check(session, *input, &oracleInfo)) {
                ++oracleStats->compactTurnFailures;
                if (oracleStats->firstFailure.empty()) {
                    oracleStats->firstFailure = "compact turn oracle API failure before input token: " + token;
                }
                return false;
            }
            if (oracleInfo.attempted) {
                ++oracleStats->compactTurnChecks;
                if (oracleInfo.handled) {
                    ++oracleStats->compactTurnHandled;
                }
                if (oracleInfo.state_checked) {
                    ++oracleStats->compactTurnStateChecks;
                }
                if (!oracleInfo.matched) {
                    ++oracleStats->compactTurnFailures;
                    if (oracleStats->firstFailure.empty()) {
                        oracleStats->firstFailure = "compact turn oracle mismatch before input token: "
                            + token
                            + compactTurnOracleMismatchSummary(oracleInfo);
                    }
                    return false;
                }
            }
        }

        ps_step_result stepResult{};
        if (*input == PS_INPUT_TICK) {
            stepResult = ps_full_state_turn(session, PS_INPUT_TICK);
        } else {
            stepResult = ps_full_state_turn(session, *input);
        }

        if (outSounds && stepResult.audio_event_count > 0 && stepResult.audio_events) {
            for (size_t i = 0; i < stepResult.audio_event_count; ++i) {
                const ps_audio_event& event = stepResult.audio_events[i];
                if (event.kind && event.kind[0] != '\0') {
                    outSounds->push_back(event.kind);
                }
            }
        }

        for (int againPass = 0; againPass < 500 && ps_full_state_pending_again(session); ++againPass) {
            stepResult = ps_full_state_turn(session, PS_INPUT_TICK);
            if (outSounds && stepResult.audio_event_count > 0 && stepResult.audio_events) {
                for (size_t i = 0; i < stepResult.audio_event_count; ++i) {
                    const ps_audio_event& event = stepResult.audio_events[i];
                    if (event.kind && event.kind[0] != '\0') {
                        outSounds->push_back(event.kind);
                    }
                }
            }
        }
    }
    return true;
}

struct TraceSnapshot {
    std::string phase;
    std::optional<int32_t> numericInput;
    std::optional<std::string> stringInput;
    int32_t currentLevelIndex = 0;
    std::optional<int32_t> currentLevelTarget;
    std::optional<std::string> loadedLevelSeed;
    bool titleScreen = false;
    bool textMode = false;
    int32_t titleMode = 0;
    int32_t titleSelection = 0;
    bool titleSelected = false;
    bool messageSelected = false;
    bool winning = false;
    bool randomStateValid = false;
    int32_t randomStateI = 0;
    int32_t randomStateJ = 0;
    std::vector<int32_t> randomStatePreviewBytes;
    std::string serializedLevel;
    std::vector<int32_t> newSounds;
};

struct TraceFile {
    std::optional<std::string> expectedSerializedLevel;
    std::vector<int32_t> expectedSounds;
    std::vector<TraceSnapshot> snapshots;
};

struct SessionSnapshot {
    int32_t currentLevelIndex = 0;
    std::optional<int32_t> currentLevelTarget;
    std::optional<std::string> loadedLevelSeed;
    bool titleScreen = false;
    bool textMode = false;
    int32_t titleMode = 0;
    int32_t titleSelection = 0;
    bool titleSelected = false;
    bool messageSelected = false;
    bool winning = false;
    bool randomStateValid = false;
    int32_t randomStateI = 0;
    int32_t randomStateJ = 0;
    std::vector<int32_t> randomStatePreviewBytes;
    std::string serializedLevel;
};

struct SimulationFixtureEntry {
    std::string name;
    std::filesystem::path irFile;
    std::optional<std::filesystem::path> traceFile;
    std::string initialSerializedLevel;
};

TraceSnapshot parseTraceSnapshot(const puzzlescript::json::Value& snapshotValue) {
    const auto& object = snapshotValue.asObject();
    TraceSnapshot snapshot;
    snapshot.phase = requireField(object, "phase").asString();
    if (const auto inputIt = object.find("input"); inputIt != object.end() && !inputIt->second.isNull()) {
        if (inputIt->second.isString()) {
            snapshot.stringInput = inputIt->second.asString();
        } else {
            snapshot.numericInput = looseAsInt(inputIt->second);
        }
    }
    snapshot.currentLevelIndex = looseAsInt(requireField(object, "current_level_index"));
    snapshot.currentLevelTarget = optionalIntField(object, "current_level_target");
    snapshot.loadedLevelSeed = optionalStringField(object, "loaded_level_seed");
    snapshot.titleScreen = requireField(object, "title_screen").asBool();
    snapshot.textMode = requireField(object, "text_mode").asBool();
    if (const auto titleMode = optionalIntField(object, "title_mode"); titleMode.has_value()) {
        snapshot.titleMode = *titleMode;
    }
    if (const auto titleSelection = optionalIntField(object, "title_selection"); titleSelection.has_value()) {
        snapshot.titleSelection = *titleSelection;
    }
    if (const auto it = object.find("title_selected"); it != object.end()) {
        snapshot.titleSelected = it->second.asBool();
    }
    if (const auto it = object.find("message_selected"); it != object.end()) {
        snapshot.messageSelected = it->second.asBool();
    }
    snapshot.winning = requireField(object, "winning").asBool();
    if (const auto it = object.find("random_state_valid"); it != object.end()) {
        snapshot.randomStateValid = it->second.asBool();
    }
    if (const auto randomStateI = optionalIntField(object, "random_state_i"); randomStateI.has_value()) {
        snapshot.randomStateI = *randomStateI;
    }
    if (const auto randomStateJ = optionalIntField(object, "random_state_j"); randomStateJ.has_value()) {
        snapshot.randomStateJ = *randomStateJ;
    }
    if (const auto it = object.find("random_state_preview_bytes"); it != object.end()) {
        snapshot.randomStatePreviewBytes = looseAsIntArray(it->second);
    }
    snapshot.serializedLevel = requireField(object, "serialized_level").asString();
    if (const auto it = object.find("new_sounds"); it != object.end()) {
        snapshot.newSounds = looseAsIntArray(it->second);
    }
    return snapshot;
}

SessionSnapshot parseSessionSnapshot(const std::string& snapshotJson, const std::string& serializedLevel) {
    const auto value = puzzlescript::json::parse(snapshotJson);
    const auto& object = value.asObject();
    SessionSnapshot snapshot;
    snapshot.currentLevelIndex = looseAsInt(requireField(object, "current_level_index"));
    snapshot.currentLevelTarget = optionalIntField(object, "current_level_target");
    snapshot.loadedLevelSeed = optionalStringField(object, "loaded_level_seed");
    snapshot.titleScreen = requireField(object, "title_screen").asBool();
    snapshot.textMode = requireField(object, "text_mode").asBool();
    if (const auto titleMode = optionalIntField(object, "title_mode"); titleMode.has_value()) {
        snapshot.titleMode = *titleMode;
    }
    if (const auto titleSelection = optionalIntField(object, "title_selection"); titleSelection.has_value()) {
        snapshot.titleSelection = *titleSelection;
    }
    if (const auto it = object.find("title_selected"); it != object.end()) {
        snapshot.titleSelected = it->second.asBool();
    }
    if (const auto it = object.find("message_selected"); it != object.end()) {
        snapshot.messageSelected = it->second.asBool();
    }
    snapshot.winning = requireField(object, "winning").asBool();
    if (const auto it = object.find("random_state_valid"); it != object.end()) {
        snapshot.randomStateValid = it->second.asBool();
    }
    if (const auto randomStateI = optionalIntField(object, "random_state_i"); randomStateI.has_value()) {
        snapshot.randomStateI = *randomStateI;
    }
    if (const auto randomStateJ = optionalIntField(object, "random_state_j"); randomStateJ.has_value()) {
        snapshot.randomStateJ = *randomStateJ;
    }
    if (const auto it = object.find("random_state_preview_bytes"); it != object.end()) {
        snapshot.randomStatePreviewBytes = looseAsIntArray(it->second);
    }
    snapshot.serializedLevel = serializedLevel;
    return snapshot;
}

std::vector<SimulationFixtureEntry> parseSimulationFixtureManifest(
    const std::filesystem::path& manifestPath,
    const std::filesystem::path& manifestDir
) {
    const auto manifestValue = puzzlescript::json::parse(readFile(manifestPath));
    const auto* fixturesValue = manifestValue.find("simulation_fixtures");
    if (!fixturesValue || !fixturesValue->isArray()) {
        throw std::runtime_error("fixtures.json is missing simulation_fixtures");
    }

    std::vector<SimulationFixtureEntry> fixtures;
    fixtures.reserve(fixturesValue->asArray().size());
    for (const auto& fixtureValue : fixturesValue->asArray()) {
        const auto& object = fixtureValue.asObject();
        SimulationFixtureEntry fixture;
        fixture.name = requireField(object, "name").asString();
        fixture.irFile = manifestDir / requireField(object, "ir_file").asString();
        fixture.initialSerializedLevel = requireField(object, "initial_serialized_level").asString();
        if (const auto traceFile = optionalStringField(object, "trace_file"); traceFile.has_value()) {
            fixture.traceFile = manifestDir / *traceFile;
        }
        fixtures.push_back(std::move(fixture));
    }
    return fixtures;
}

std::vector<TraceSnapshot> loadTraceSnapshotsFromJsonText(const std::string& jsonText) {
    const auto root = puzzlescript::json::parse(jsonText);
    const auto* traceValue = root.find("trace");
    if (!traceValue || !traceValue->isObject()) {
        throw std::runtime_error("Trace file is missing trace object");
    }
    const auto* snapshotsValue = traceValue->find("snapshots");
    if (!snapshotsValue || !snapshotsValue->isArray()) {
        throw std::runtime_error("Trace file is missing snapshots");
    }

    std::vector<TraceSnapshot> snapshots;
    snapshots.reserve(snapshotsValue->asArray().size());
    for (const auto& snapshotValue : snapshotsValue->asArray()) {
        snapshots.push_back(parseTraceSnapshot(snapshotValue));
    }

    return snapshots;
}

std::vector<TraceSnapshot> loadTraceSnapshots(const std::filesystem::path& path) {
    return loadTraceSnapshotsFromJsonText(readFile(path));
}

TraceFile loadTraceFileFromJsonText(const std::string& jsonText) {
    const auto root = puzzlescript::json::parse(jsonText);
    TraceFile result;
    if (const auto* expectedSerialized = root.find("expected_serialized_level"); expectedSerialized && expectedSerialized->isString()) {
        result.expectedSerializedLevel = expectedSerialized->asString();
    }
    if (const auto* expectedSounds = root.find("expected_sounds"); expectedSounds && expectedSounds->isArray()) {
        result.expectedSounds = looseAsIntArray(*expectedSounds);
    }
    result.snapshots = loadTraceSnapshotsFromJsonText(jsonText);
    return result;
}

TraceFile loadTraceFile(const std::filesystem::path& path) {
    return loadTraceFileFromJsonText(readFile(path));
}

size_t traceSnapshotProgressInterval() {
    const char* value = std::getenv("PS_TRACE_PROGRESS");
    if (value == nullptr || value[0] == '\0') {
        return 0;
    }
    char* end = nullptr;
    const unsigned long long parsed = std::strtoull(value, &end, 10);
    if (end == value || parsed == 0) {
        return 0;
    }
    return static_cast<size_t>(parsed);
}

bool loadGameFromJsonText(const std::string& json, ps_game** outGame) {
    ps_error* error = nullptr;
    if (!ps_load_ir_json(json.data(), json.size(), outGame, &error)) {
        std::cerr << ps_error_message(error) << "\n";
        ps_free_error(error);
        return false;
    }
    return true;
}

bool loadGameFromSourceText(const std::string& sourceText, ps_game** outGame) {
    if (!outGame) {
        return false;
    }
    ps_compile_result* result = nullptr;
    if (!ps_compile_source(sourceText.data(), sourceText.size(), &result) || result == nullptr) {
        if (result) {
            const ps_error* error = ps_compile_result_error(result);
            if (error) {
                std::cerr << ps_error_message(error) << "\n";
                ps_free_error(const_cast<ps_error*>(error));
            }
            ps_free_compile_result(result);
        }
        return false;
    }
    const ps_game* game = ps_compile_result_game(result);
    if (!game) {
        const ps_error* error = ps_compile_result_error(result);
        if (error) {
            std::cerr << ps_error_message(error) << "\n";
            ps_free_error(const_cast<ps_error*>(error));
        }
        ps_free_compile_result(result);
        return false;
    }
    // ps_compile_result_game() returns a newly-allocated ps_game wrapper.
    *outGame = const_cast<ps_game*>(game);
    ps_free_compile_result(result);
    return true;
}

bool loadGameFromSourceFile(const std::filesystem::path& path, ps_game** outGame) {
    return loadGameFromSourceText(readFile(path) + "\n", outGame);
}

bool loadGameFromFile(const std::filesystem::path& path, ps_game** outGame) {
    return loadGameFromJsonText(readFile(path), outGame);
}

struct PsGameCache {
    std::unordered_map<std::string, ps_game*> games;

    PsGameCache() = default;

    ~PsGameCache() {
        for (auto& entry : games) {
            if (entry.second != nullptr) {
                ps_free_game(entry.second);
            }
        }
    }

    PsGameCache(const PsGameCache&) = delete;
    PsGameCache& operator=(const PsGameCache&) = delete;

    bool has(const std::filesystem::path& irPath) const {
        const std::string key = irPath.lexically_normal().string();
        return games.find(key) != games.end();
    }

    ps_game* acquire(const std::filesystem::path& irPath) {
        const std::string key = irPath.lexically_normal().string();
        const auto existing = games.find(key);
        if (existing != games.end()) {
            return existing->second;
        }
        ps_game* game = nullptr;
        if (!loadGameFromFile(irPath, &game)) {
            return nullptr;
        }
        games.emplace(key, game);
        return game;
    }
};

std::string shellEscape(const std::string& value) {
    std::string escaped = "'";
    for (const char ch : value) {
        if (ch == '\'') {
            escaped += "'\"'\"'";
        } else {
            escaped.push_back(ch);
        }
    }
    escaped.push_back('\'');
    return escaped;
}

std::string runNodeScriptAndCaptureStdout(
    const std::string& scriptPath,
    const std::filesystem::path& sourcePath,
    const std::vector<std::string>& args
) {
    std::ostringstream command;
    command << shellEscape(PS_NODE_EXECUTABLE) << " "
            << shellEscape(scriptPath) << " "
            << shellEscape(sourcePath.string());
    for (const auto& arg : args) {
        command << " " << shellEscape(arg);
    }

    std::string output;
    std::array<char, 4096> buffer{};
    std::unique_ptr<FILE, decltype(&pclose)> pipe(popen(command.str().c_str(), "r"), pclose);
    if (!pipe) {
        throw std::runtime_error("Failed to launch JS IR exporter");
    }
    while (fgets(buffer.data(), static_cast<int>(buffer.size()), pipe.get()) != nullptr) {
        output.append(buffer.data());
    }
    const int status = pclose(pipe.release());
    if (status != 0) {
        throw std::runtime_error("JS IR exporter failed");
    }
    return output;
}

std::string runIrExporterAndCaptureJson(const std::filesystem::path& sourcePath, const std::vector<std::string>& args) {
    return runNodeScriptAndCaptureStdout(PS_EXPORT_IR_SCRIPT, sourcePath, args);
}

std::string runTraceExporterAndCaptureJson(const std::filesystem::path& sourcePath, const std::vector<std::string>& args) {
    return runNodeScriptAndCaptureStdout(PS_EXPORT_TRACE_SCRIPT, sourcePath, args);
}

int printSession(ps_full_state* session) {
    char* serialized = ps_full_state_serialize_test_string(session);
    char* snapshot = ps_full_state_export_snapshot(session);
    const uint64_t hash64 = ps_full_state_hash64(session);
    const ps_hash128 hash128 = ps_full_state_hash128(session);

    std::cout << serialized;
    std::cout << "hash64=" << hash64 << "\n";
    std::cout << "hash128=" << hash128.lo << ":" << hash128.hi << "\n";
    std::cout << snapshot << "\n";

    ps_string_free(snapshot);
    ps_string_free(serialized);
    return 0;
}

int runCommandForGame(ps_game* game) {
    ps_full_state* session = nullptr;
    ps_error* error = nullptr;
    if (!ps_full_state_create(game, &session, &error)) {
        std::cerr << ps_error_message(error) << "\n";
        ps_free_error(error);
        return 1;
    }

    const int result = printSession(session);
    ps_full_state_destroy(session);
    return result;
}

int benchCommandForGame(
    ps_game* game,
    uint32_t iterations,
    uint32_t threads,
    const std::optional<std::string>& loadedLevelSeed = std::nullopt
) {
    ps_full_state* session = nullptr;
    ps_error* error = nullptr;
    if (!sessionCreateForGame(game, loadedLevelSeed, &session, &error)) {
        std::cerr << ps_error_message(error) << "\n";
        ps_free_error(error);
        return 1;
    }

    ps_benchmark_result result{};
    if (!ps_benchmark_full_state_clone_hash(session, iterations, threads, &result, &error)) {
        std::cerr << ps_error_message(error) << "\n";
        ps_free_error(error);
        ps_full_state_destroy(session);
        return 1;
    }

    std::cout << "iterations=" << result.iterations
              << " threads=" << result.threads
              << " elapsed_seconds=" << result.elapsed_seconds
              << " iterations_per_second=" << result.iterations_per_second
              << " hash_accumulator=" << result.hash_accumulator
              << "\n";

    ps_full_state_destroy(session);
    return 0;
}

std::optional<ps_input> parseInputToken(const std::string& token) {
    if (token == "up") {
        return PS_INPUT_UP;
    }
    if (token == "left") {
        return PS_INPUT_LEFT;
    }
    if (token == "down") {
        return PS_INPUT_DOWN;
    }
    if (token == "right") {
        return PS_INPUT_RIGHT;
    }
    if (token == "action") {
        return PS_INPUT_ACTION;
    }
    if (token == "tick") {
        return PS_INPUT_TICK;
    }
    try {
        size_t consumed = 0;
        int32_t inputValue = std::stoi(token, &consumed);
        if (consumed != token.size()) {
            return std::nullopt;
        }
        if (inputValue < 0) {
            inputValue = 0;
        }
        if (inputValue > static_cast<int32_t>(PS_INPUT_TICK)) {
            inputValue = static_cast<int32_t>(PS_INPUT_TICK);
        }
        return static_cast<ps_input>(inputValue);
    } catch (...) {
        return std::nullopt;
    }
    return std::nullopt;
}

int stepCommandForGame(
    ps_game* game,
    const std::vector<std::string>& inputTokens,
    const std::optional<std::string>& loadedLevelSeed = std::nullopt
) {
    ps_full_state* session = nullptr;
    ps_error* error = nullptr;
    if (!sessionCreateForGame(game, loadedLevelSeed, &session, &error)) {
        std::cerr << ps_error_message(error) << "\n";
        ps_free_error(error);
        return 1;
    }

    for (size_t index = 0; index < inputTokens.size(); ++index) {
        const auto& token = inputTokens[index];
        ps_step_result result{};
        if (token == "undo") {
            if (!ps_full_state_undo(session)) {
                std::cerr << "step[" << index << "] undo failed\n";
                ps_full_state_destroy(session);
                return 1;
            }
        } else if (token == "restart") {
            if (!ps_full_state_restart(session)) {
                std::cerr << "step[" << index << "] restart failed\n";
                ps_full_state_destroy(session);
                return 1;
            }
        } else {
            const auto input = parseInputToken(token);
            if (!input.has_value()) {
                std::cerr << "Unsupported step token: " << token << "\n";
                ps_full_state_destroy(session);
                return 1;
            }
            result = (*input == PS_INPUT_TICK) ? ps_full_state_turn(session, PS_INPUT_TICK) : ps_full_state_turn(session, *input);
        }
        std::cout << "step[" << index << "] token=" << token
                  << " changed=" << (result.changed ? 1 : 0)
                  << " transitioned=" << (result.transitioned ? 1 : 0)
                  << " won=" << (result.won ? 1 : 0)
                  << " audio_events=" << result.audio_event_count
                  << "\n";
    }

    const int result = printSession(session);
    ps_full_state_destroy(session);
    return result;
}

bool compareSnapshot(const TraceSnapshot& expected, ps_full_state* session, const ps_step_result* stepResult, size_t snapshotIndex, std::ostream& stream) {
    ps_full_state_status_info status{};
    ps_full_state_status(session, &status);
    char* serialized = ps_full_state_serialize_test_string(session);
    const std::string actualSerialized = serialized ? serialized : "";
    ps_string_free(serialized);
    char* snapshotJson = ps_full_state_export_snapshot(session);
    const std::string actualSnapshotJson = snapshotJson ? snapshotJson : "";
    ps_string_free(snapshotJson);
    const SessionSnapshot actualSnapshot = parseSessionSnapshot(actualSnapshotJson, actualSerialized);

    bool ok = true;
    if (actualSnapshot.serializedLevel != expected.serializedLevel) {
        stream << "snapshot[" << snapshotIndex << "] serialized level mismatch\n";
        stream << "expected_serialized_level:\n" << expected.serializedLevel;
        if (!expected.serializedLevel.empty() && expected.serializedLevel.back() != '\n') {
            stream << "\n";
        }
        stream << "actual_serialized_level:\n" << actualSnapshot.serializedLevel;
        if (!actualSnapshot.serializedLevel.empty() && actualSnapshot.serializedLevel.back() != '\n') {
            stream << "\n";
        }
        stream << "actual_session_snapshot:\n" << actualSnapshotJson << "\n";
        ok = false;
    }
    if (status.current_level_index != expected.currentLevelIndex) {
        stream << "snapshot[" << snapshotIndex << "] current_level_index mismatch: actual="
               << status.current_level_index << " expected=" << expected.currentLevelIndex << "\n";
        ok = false;
    }
    if (status.has_current_level_target != expected.currentLevelTarget.has_value()
        || (expected.currentLevelTarget.has_value() && status.current_level_target != *expected.currentLevelTarget)) {
        stream << "snapshot[" << snapshotIndex << "] current_level_target mismatch\n";
        ok = false;
    }
    if (status.title_screen != expected.titleScreen) {
        stream << "snapshot[" << snapshotIndex << "] title_screen mismatch\n";
        ok = false;
    }
    if (status.text_mode != expected.textMode) {
        stream << "snapshot[" << snapshotIndex << "] text_mode mismatch\n";
        ok = false;
    }
    if (status.title_mode != expected.titleMode) {
        stream << "snapshot[" << snapshotIndex << "] title_mode mismatch: actual="
               << status.title_mode << " expected=" << expected.titleMode << "\n";
        ok = false;
    }
    if (status.title_selection != expected.titleSelection) {
        stream << "snapshot[" << snapshotIndex << "] title_selection mismatch: actual="
               << status.title_selection << " expected=" << expected.titleSelection << "\n";
        ok = false;
    }
    if (status.title_selected != expected.titleSelected) {
        stream << "snapshot[" << snapshotIndex << "] title_selected mismatch\n";
        ok = false;
    }
    if (status.message_selected != expected.messageSelected) {
        stream << "snapshot[" << snapshotIndex << "] message_selected mismatch\n";
        ok = false;
    }
    if (status.winning != expected.winning) {
        stream << "snapshot[" << snapshotIndex << "] winning mismatch\n";
        ok = false;
    }
    const bool skipRandomStateComparison = expected.loadedLevelSeed.has_value()
        && actualSnapshot.loadedLevelSeed.has_value()
        && *expected.loadedLevelSeed != *actualSnapshot.loadedLevelSeed;
    if (!skipRandomStateComparison) {
        if (actualSnapshot.randomStateValid != expected.randomStateValid) {
            stream << "snapshot[" << snapshotIndex << "] random_state_valid mismatch: actual="
                   << (actualSnapshot.randomStateValid ? 1 : 0) << " expected=" << (expected.randomStateValid ? 1 : 0) << "\n";
            ok = false;
        }
        if (actualSnapshot.randomStateI != expected.randomStateI) {
            stream << "snapshot[" << snapshotIndex << "] random_state_i mismatch: actual="
                   << actualSnapshot.randomStateI << " expected=" << expected.randomStateI << "\n";
            ok = false;
        }
        if (actualSnapshot.randomStateJ != expected.randomStateJ) {
            stream << "snapshot[" << snapshotIndex << "] random_state_j mismatch: actual="
                   << actualSnapshot.randomStateJ << " expected=" << expected.randomStateJ << "\n";
            ok = false;
        }
        if (actualSnapshot.randomStatePreviewBytes != expected.randomStatePreviewBytes) {
            stream << "snapshot[" << snapshotIndex << "] random_state_preview_bytes mismatch: actual=[";
            appendIntList(stream, actualSnapshot.randomStatePreviewBytes);
            stream << "] expected=[";
            appendIntList(stream, expected.randomStatePreviewBytes);
            stream << "]\n";
            ok = false;
        }
    }
    const bool skipTraceAudio = std::getenv("PS_TRACE_IGNORE_AUDIO") != nullptr;
    if (!skipTraceAudio && stepResult) {
        if (stepResult->audio_event_count != expected.newSounds.size()) {
            stream << "snapshot[" << snapshotIndex << "] audio event count mismatch: actual="
                   << stepResult->audio_event_count << " expected=" << expected.newSounds.size() << "\n";
            stream << "actual_audio_seeds=[";
            for (size_t soundIndex = 0; soundIndex < stepResult->audio_event_count; ++soundIndex) {
                if (soundIndex > 0) {
                    stream << ",";
                }
                stream << stepResult->audio_events[soundIndex].seed;
            }
            stream << "] expected_audio_seeds=[";
            for (size_t soundIndex = 0; soundIndex < expected.newSounds.size(); ++soundIndex) {
                if (soundIndex > 0) {
                    stream << ",";
                }
                stream << expected.newSounds[soundIndex];
            }
            stream << "]\n";
            ok = false;
        } else {
            for (size_t soundIndex = 0; soundIndex < expected.newSounds.size(); ++soundIndex) {
                if (stepResult->audio_events[soundIndex].seed != expected.newSounds[soundIndex]) {
                    stream << "snapshot[" << snapshotIndex << "] audio seed mismatch at index "
                           << soundIndex << ": actual=" << stepResult->audio_events[soundIndex].seed
                           << " expected=" << expected.newSounds[soundIndex] << "\n";
                    ok = false;
                }
            }
        }
    } else if (!skipTraceAudio && !expected.newSounds.empty()) {
        stream << "snapshot[" << snapshotIndex << "] expected audio events but no step result was provided\n";
        ok = false;
    }
    return ok;
}

int diffTraceAgainstSnapshots(
    ps_game* game,
    const std::vector<TraceSnapshot>& snapshots,
    std::ostream& errorStream,
    bool printSuccessSummary,
    const std::optional<std::string>& loadedLevelSeed = std::nullopt,
    const std::optional<int32_t>& levelToLoad = std::nullopt
) {
    ps_full_state* session = nullptr;
    ps_error* error = nullptr;
    if (!sessionCreateForGame(game, loadedLevelSeed, &session, &error)) {
        errorStream << ps_error_message(error) << "\n";
        ps_free_error(error);
        return 1;
    }
    std::unique_ptr<ps_full_state, decltype(&ps_full_state_destroy)> sessionHolder(session, ps_full_state_destroy);
    ps_full_state_set_unit_testing(session, true);

    if (levelToLoad.has_value()) {
        if (!ps_full_state_load_level(session, *levelToLoad, &error)) {
            errorStream << ps_error_message(error) << "\n";
            ps_free_error(error);
            return 1;
        }
    }

    if (snapshots.empty()) {
        errorStream << "Trace has no snapshots\n";
        return 1;
    }
    if (!compareSnapshot(snapshots.front(), session, nullptr, 0, errorStream)) {
        return 1;
    }

    for (size_t index = 1; index < snapshots.size(); ++index) {
        const size_t progressInterval = traceSnapshotProgressInterval();
        if (progressInterval > 0 && (index % progressInterval) == 0) {
            errorStream << "trace_snapshot_progress index=" << index
                        << " total=" << snapshots.size()
                        << " phase=" << snapshots[index].phase
                        << "\n"
                        << std::flush;
        }
        const auto& snapshot = snapshots[index];
        ps_step_result stepResult{};
        if (snapshot.phase == "again") {
            stepResult = ps_full_state_turn(session, PS_INPUT_TICK);
        } else if (snapshot.numericInput.has_value()) {
            stepResult = ps_full_state_turn(session, static_cast<ps_input>(*snapshot.numericInput));
        } else if (snapshot.stringInput.has_value()) {
            if (*snapshot.stringInput == "tick") {
                stepResult = ps_full_state_turn(session, PS_INPUT_TICK);
            } else if (*snapshot.stringInput == "restart") {
                if (!ps_full_state_restart(session)) {
                    errorStream << "Restart failed at snapshot[" << index << "]\n";
                    return 1;
                }
            } else if (*snapshot.stringInput == "undo") {
                if (!ps_full_state_undo(session)) {
                    errorStream << "Undo failed at snapshot[" << index << "]\n";
                    return 1;
                }
            } else {
                errorStream << "Unsupported trace input token: " << *snapshot.stringInput << "\n";
                return 1;
            }
        } else {
            errorStream << "Snapshot[" << index << "] has no replayable input token\n";
            return 1;
        }

        if (!compareSnapshot(snapshot, session, &stepResult, index, errorStream)) {
            return 1;
        }
    }

    if (printSuccessSummary) {
        std::cout << "trace_diff_passed snapshots=" << snapshots.size() << "\n";
    }
    return 0;
}

int diffTraceCommand(const std::string& irPath, const std::string& tracePath) {
    ps_game* game = nullptr;
    if (!loadGameFromFile(irPath, &game)) {
        return 1;
    }
    const int result = diffTraceAgainstSnapshots(game, loadTraceSnapshots(tracePath), std::cerr, true);
    ps_free_game(game);
    return result;
}

int checkTraceAgainstSnapshots(ps_game* game, const TraceFile& traceFile, std::ostream& errorStream, bool printSuccessSummary) {
    ps_full_state* session = nullptr;
    ps_error* error = nullptr;
    if (!ps_full_state_create(game, &session, &error)) {
        errorStream << ps_error_message(error) << "\n";
        ps_free_error(error);
        return 1;
    }
    std::unique_ptr<ps_full_state, decltype(&ps_full_state_destroy)> sessionHolder(session, ps_full_state_destroy);
    ps_full_state_set_unit_testing(session, true);

    const auto& snapshots = traceFile.snapshots;
    if (snapshots.empty()) {
        errorStream << "Trace has no snapshots\n";
        return 1;
    }

    // Fast path: just replay inputs to reach the final state without exporting/parsing per-snapshot JSON.
    std::vector<int32_t> observedSounds;
    if (!traceFile.expectedSounds.empty()) {
        observedSounds.reserve(traceFile.expectedSounds.size());
    }

    for (size_t index = 1; index < snapshots.size(); ++index) {
        const auto& snapshot = snapshots[index];
        ps_step_result stepResult{};
        if (snapshot.phase == "again") {
            stepResult = ps_full_state_turn(session, PS_INPUT_TICK);
        } else if (snapshot.numericInput.has_value()) {
            stepResult = ps_full_state_turn(session, static_cast<ps_input>(*snapshot.numericInput));
        } else if (snapshot.stringInput.has_value()) {
            if (*snapshot.stringInput == "tick") {
                stepResult = ps_full_state_turn(session, PS_INPUT_TICK);
            } else if (*snapshot.stringInput == "restart") {
                if (!ps_full_state_restart(session)) {
                    errorStream << "Restart failed at snapshot[" << index << "]\n";
                    return 1;
                }
            } else if (*snapshot.stringInput == "undo") {
                if (!ps_full_state_undo(session)) {
                    errorStream << "Undo failed at snapshot[" << index << "]\n";
                    return 1;
                }
            } else {
                errorStream << "Unsupported trace input token: " << *snapshot.stringInput << "\n";
                return 1;
            }
        } else {
            errorStream << "Snapshot[" << index << "] has no replayable input token\n";
            return 1;
        }

        if (!traceFile.expectedSounds.empty() && stepResult.audio_event_count > 0 && stepResult.audio_events != nullptr) {
            for (size_t soundIndex = 0; soundIndex < stepResult.audio_event_count; ++soundIndex) {
                observedSounds.push_back(stepResult.audio_events[soundIndex].seed);
            }
        }
    }

    const auto& expectedFinal = snapshots.back().serializedLevel;
    char* serialized = ps_full_state_serialize_test_string(session);
    const std::string actualFinal = serialized ? serialized : "";
    ps_string_free(serialized);
    if (actualFinal != expectedFinal) {
        errorStream << "final serialized level mismatch\n";
        return 1;
    }

    if (!traceFile.expectedSounds.empty() && observedSounds != traceFile.expectedSounds) {
        errorStream << "final expected_sounds mismatch\n";
        return 1;
    }

    if (printSuccessSummary) {
        std::cout << "trace_check_passed snapshots=" << snapshots.size() << "\n";
    }
    return 0;
}

bool replayTraceInputsOnly(ps_full_state* session, const TraceFile& traceFile, std::ostream& errorStream, size_t& replayedSteps) {
    const auto& snapshots = traceFile.snapshots;
    if (snapshots.empty()) {
        errorStream << "Trace has no snapshots\n";
        return false;
    }

    for (size_t index = 1; index < snapshots.size(); ++index) {
        const auto& snapshot = snapshots[index];
        if (snapshot.phase == "again") {
            (void)ps_full_state_turn(session, PS_INPUT_TICK);
            ++replayedSteps;
        } else if (snapshot.numericInput.has_value()) {
            (void)ps_full_state_turn(session, static_cast<ps_input>(*snapshot.numericInput));
            ++replayedSteps;
        } else if (snapshot.stringInput.has_value()) {
            if (*snapshot.stringInput == "tick") {
                (void)ps_full_state_turn(session, PS_INPUT_TICK);
                ++replayedSteps;
            } else if (*snapshot.stringInput == "restart") {
                if (!ps_full_state_restart(session)) {
                    errorStream << "Restart failed at snapshot[" << index << "]\n";
                    return false;
                }
                ++replayedSteps;
            } else if (*snapshot.stringInput == "undo") {
                if (!ps_full_state_undo(session)) {
                    errorStream << "Undo failed at snapshot[" << index << "]\n";
                    return false;
                }
                ++replayedSteps;
            } else {
                errorStream << "Unsupported trace input token: " << *snapshot.stringInput << "\n";
                return false;
            }
        } else {
            errorStream << "Snapshot[" << index << "] has no replayable input token\n";
            return false;
        }
    }
    return true;
}

int checkTraceCommand(const std::string& irPath, const std::string& tracePath) {
    ps_game* game = nullptr;
    if (!loadGameFromFile(irPath, &game)) {
        return 1;
    }
    const int result = checkTraceAgainstSnapshots(game, loadTraceFile(tracePath), std::cerr, true);
    ps_free_game(game);
    return result;
}

bool runPreparedSerializedLevelCheck(
    ps_game* game,
    const std::string& expectedSerialized,
    const std::string& name,
    bool quiet,
    bool profileTimers,
    int64_t& profileSessionCreateUs,
    int64_t& profileSerializePreparedUs
) {
    ps_full_state* session = nullptr;
    ps_error* error = nullptr;
    const auto sessionStart = std::chrono::steady_clock::now();
    if (!ps_full_state_create(game, &session, &error)) {
        if (!quiet) {
            std::cerr << name << ": " << ps_error_message(error) << "\n";
        }
        ps_free_error(error);
        if (profileTimers) {
            profileSessionCreateUs += std::chrono::duration_cast<std::chrono::microseconds>(
                                           std::chrono::steady_clock::now() - sessionStart)
                                           .count();
        }
        return false;
    }
    if (profileTimers) {
        profileSessionCreateUs += std::chrono::duration_cast<std::chrono::microseconds>(
                                       std::chrono::steady_clock::now() - sessionStart)
                                       .count();
    }

    const auto serializeStart = std::chrono::steady_clock::now();
    char* serialized = ps_full_state_serialize_test_string(session);
    const std::string actual = serialized ? serialized : "";
    if (profileTimers) {
        profileSerializePreparedUs += std::chrono::duration_cast<std::chrono::microseconds>(
                                           std::chrono::steady_clock::now() - serializeStart)
                                           .count();
    }
    ps_string_free(serialized);
    ps_full_state_destroy(session);

    if (actual == expectedSerialized) {
        return true;
    }
    if (!quiet) {
        std::cerr << name << ": prepared session mismatch\n";
    }
    return false;
}

int checkTraceSweepCommand(const std::string& manifestPath, int argc, char** argv) {
    bool quiet = false;
    bool allowFailures = false;
    bool profileTimers = false;
    size_t progressEvery = 0;

    for (int argIndex = 0; argIndex < argc; ++argIndex) {
        const std::string arg = argv[argIndex];
        if (arg == "--quiet") {
            quiet = true;
        } else if (arg == "--allow-failures") {
            allowFailures = true;
        } else if (arg == "--profile-timers") {
            profileTimers = true;
        } else if (arg == "--progress-every" && argIndex + 1 < argc) {
            progressEvery = static_cast<size_t>(std::stoull(argv[++argIndex]));
        } else {
            throw std::runtime_error("Unsupported JS parity data check argument: " + arg);
        }
    }

    const auto manifestDir = std::filesystem::path(manifestPath).parent_path();
    size_t preparedPassed = 0;
    size_t preparedFailed = 0;
    size_t traceChecked = 0;
    size_t tracePassed = 0;
    size_t traceFailed = 0;
    size_t traceTimedOut = 0;
    size_t traceFastPassed = 0;
    size_t traceDetailedRuns = 0;

    int64_t profileGameReuseUs = 0;
    int64_t profileGameLoadUs = 0;
    size_t profileGamesReused = 0;
    size_t profileGamesLoaded = 0;
    int64_t profilePreparedSessionUs = 0;
    int64_t profilePreparedSerializeUs = 0;
    int64_t profileTraceParseUs = 0;
    int64_t profileFastCheckUs = 0;
    int64_t profileDiffUs = 0;
    const auto sweepWallStart = std::chrono::steady_clock::now();

    try {
        PsGameCache cache;
        const auto fixtures = parseSimulationFixtureManifest(manifestPath, manifestDir);
        for (const auto& fixture : fixtures) {
            const std::string& name = fixture.name;
            const auto& irPath = fixture.irFile;
            const bool hasTrace = fixture.traceFile.has_value();

            const bool gameCached = cache.has(irPath);
            const auto gameAcquireStart = std::chrono::steady_clock::now();
            ps_game* game = cache.acquire(irPath);
            const auto gameAcquireUs = std::chrono::duration_cast<std::chrono::microseconds>(
                                           std::chrono::steady_clock::now() - gameAcquireStart)
                                           .count();
            if (profileTimers) {
                if (gameCached) {
                    ++profileGamesReused;
                    profileGameReuseUs += gameAcquireUs;
                } else {
                    ++profileGamesLoaded;
                    profileGameLoadUs += gameAcquireUs;
                }
            }

            if (game == nullptr) {
                ++preparedFailed;
                if (hasTrace) {
                    ++traceFailed;
                }
                if (!quiet) {
                    std::cerr << name << ": failed to load IR\n";
                }
                if (hasTrace) {
                    ++traceChecked;
                    const auto fastMs = static_cast<int64_t>(0);
                    std::cerr << "trace_case index=" << traceChecked << " name=" << name << " mode=fast outcome=failed elapsed_ms=" << fastMs
                              << "\n";
                    if (progressEvery > 0 && (traceChecked % progressEvery) == 0) {
                        std::cerr << "trace_progress checked=" << traceChecked << " passed=" << tracePassed << " failed=" << traceFailed
                                  << " timed_out=" << traceTimedOut << " current_case=" << name << " outcome=failed elapsed_ms=" << fastMs
                                  << "\n";
                    }
                }
                continue;
            }

            if (runPreparedSerializedLevelCheck(
                    game,
                    fixture.initialSerializedLevel,
                    name,
                    quiet,
                    profileTimers,
                    profilePreparedSessionUs,
                    profilePreparedSerializeUs
                )) {
                ++preparedPassed;
            } else {
                ++preparedFailed;
            }

            if (!hasTrace) {
                continue;
            }

            ++traceChecked;
            const auto& tracePath = *fixture.traceFile;
            const auto fastStarted = std::chrono::steady_clock::now();

            TraceFile traceFile;
            try {
                const auto traceParseStart = std::chrono::steady_clock::now();
                traceFile = loadTraceFile(tracePath);
                if (profileTimers) {
                    profileTraceParseUs += std::chrono::duration_cast<std::chrono::microseconds>(
                                               std::chrono::steady_clock::now() - traceParseStart)
                                               .count();
                }
            } catch (const std::exception& error) {
                ++traceFailed;
                const auto fastMs = std::chrono::duration_cast<std::chrono::milliseconds>(
                                        std::chrono::steady_clock::now() - fastStarted)
                                        .count();
                if (!quiet) {
                    std::cerr << name << ": " << error.what() << "\n";
                }
                std::cerr << "trace_case index=" << traceChecked << " name=" << name << " mode=fast outcome=failed elapsed_ms=" << fastMs
                          << "\n";
                if (progressEvery > 0 && (traceChecked % progressEvery) == 0) {
                    std::cerr << "trace_progress checked=" << traceChecked << " passed=" << tracePassed << " failed=" << traceFailed
                              << " timed_out=" << traceTimedOut << " current_case=" << name << " outcome=failed elapsed_ms=" << fastMs
                              << "\n";
                }
                continue;
            }

            std::ostringstream fastErrors;
            const auto fastCheckStart = std::chrono::steady_clock::now();
            const int fastResult = checkTraceAgainstSnapshots(game, traceFile, fastErrors, false);
            if (profileTimers) {
                profileFastCheckUs += std::chrono::duration_cast<std::chrono::microseconds>(
                                          std::chrono::steady_clock::now() - fastCheckStart)
                                          .count();
            }
            const auto fastMs = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now() - fastStarted)
                                    .count();

            std::string mode = "fast";
            std::string outcome = "passed";
            int64_t elapsedMs = fastMs;

            if (fastResult == 0) {
                ++tracePassed;
                ++traceFastPassed;
            } else {
                mode = "detailed";
                ++traceDetailedRuns;
                const auto detailedStarted = std::chrono::steady_clock::now();
                std::ostringstream diffErrors;
                const int detailedResult = diffTraceAgainstSnapshots(game, traceFile.snapshots, diffErrors, false);
                if (profileTimers) {
                    profileDiffUs += std::chrono::duration_cast<std::chrono::microseconds>(
                                         std::chrono::steady_clock::now() - detailedStarted)
                                         .count();
                }
                elapsedMs = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now() - detailedStarted)
                                .count();
                if (detailedResult == 0) {
                    ++tracePassed;
                } else {
                    ++traceFailed;
                    outcome = "failed";
                }
                if (!quiet) {
                    std::cerr << fastErrors.str();
                    std::cerr << diffErrors.str();
                }
            }

            std::cerr << "trace_case index=" << traceChecked << " name=" << name << " mode=" << mode << " outcome=" << outcome << " elapsed_ms="
                      << elapsedMs;
            if (mode == "detailed") {
                std::cerr << " fast_elapsed_ms=" << fastMs;
            }
            std::cerr << "\n";

            if (progressEvery > 0 && (traceChecked % progressEvery) == 0) {
                std::cerr << "trace_progress checked=" << traceChecked << " passed=" << tracePassed << " failed=" << traceFailed
                          << " timed_out=" << traceTimedOut << " current_case=" << name << " outcome=" << outcome << " elapsed_ms=" << elapsedMs
                          << "\n";
            }
        }
    } catch (const std::exception& error) {
        std::cerr << error.what() << "\n";
        return 1;
    }

    const size_t simulationFixtureCount = preparedPassed + preparedFailed;
    std::cout << "test_cases_checked=" << simulationFixtureCount << " prepared_session_checks_passed=" << preparedPassed
              << " prepared_session_checks_failed=" << preparedFailed << " trace_replay_checked=" << traceChecked << " trace_replay_passed="
              << tracePassed << " trace_replay_failed=" << traceFailed << " trace_replay_timed_out=" << traceTimedOut
              << " trace_fast_passed=" << traceFastPassed << " trace_detailed_runs=" << traceDetailedRuns << "\n";

    if (profileTimers) {
        const auto sweepWallUs = std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::steady_clock::now() - sweepWallStart)
                                     .count();
        const auto usToMs = [](int64_t microseconds) -> int64_t {
            return (microseconds + 500) / 1000;
        };
        std::cerr << "native_trace_suite_profile test_cases=" << simulationFixtureCount << " saved_replays=" << traceChecked
                  << " wall_ms=" << usToMs(sweepWallUs) << " games_reused=" << profileGamesReused << " games_loaded=" << profileGamesLoaded
                  << " game_reuse_ms=" << usToMs(profileGameReuseUs) << " game_load_ms=" << usToMs(profileGameLoadUs)
                  << " prepared_session_create_ms=" << usToMs(profilePreparedSessionUs)
                  << " prepared_serialize_ms=" << usToMs(profilePreparedSerializeUs) << " trace_json_parse_ms=" << usToMs(profileTraceParseUs)
                  << " fast_replay_ms=" << usToMs(profileFastCheckUs) << " detailed_diff_ms=" << usToMs(profileDiffUs) << "\n";
    }

    const bool preparedOk = preparedFailed == 0;
    const bool traceOk = allowFailures || traceFailed == 0;
    (void)traceTimedOut;
    return (preparedOk && traceOk) ? 0 : 1;
}

int profileSimulationsCommand(const std::string& manifestPath, int argc, char** argv) {
    bool quiet = false;
    bool profileTimers = false;
    size_t repeat = 1;

    for (int argIndex = 0; argIndex < argc; ++argIndex) {
        const std::string arg = argv[argIndex];
        if (arg == "--quiet") {
            quiet = true;
        } else if (arg == "--profile-timers") {
            profileTimers = true;
        } else if (arg == "--repeat" && argIndex + 1 < argc) {
            repeat = std::max<size_t>(1, static_cast<size_t>(std::stoull(argv[++argIndex])));
        } else {
            throw std::runtime_error("Unsupported simulation profile argument: " + arg);
        }
    }

    const auto manifestDir = std::filesystem::path(manifestPath).parent_path();
    size_t casesChecked = 0;
    size_t casesFailed = 0;
    size_t replayRuns = 0;
    size_t replayedSteps = 0;

    int64_t profileGameReuseUs = 0;
    int64_t profileGameLoadUs = 0;
    size_t profileGamesReused = 0;
    size_t profileGamesLoaded = 0;
    int64_t profileTraceParseUs = 0;
    int64_t profileSessionCreateUs = 0;
    int64_t profileReplayUs = 0;
    const auto wallStart = std::chrono::steady_clock::now();

    try {
        PsGameCache cache;
        const auto fixtures = parseSimulationFixtureManifest(manifestPath, manifestDir);
        for (const auto& fixture : fixtures) {
            if (!fixture.traceFile.has_value()) {
                continue;
            }
            ++casesChecked;

            const bool gameCached = cache.has(fixture.irFile);
            const auto gameAcquireStart = std::chrono::steady_clock::now();
            ps_game* game = cache.acquire(fixture.irFile);
            const auto gameAcquireUs = std::chrono::duration_cast<std::chrono::microseconds>(
                                           std::chrono::steady_clock::now() - gameAcquireStart)
                                           .count();
            if (profileTimers) {
                if (gameCached) {
                    ++profileGamesReused;
                    profileGameReuseUs += gameAcquireUs;
                } else {
                    ++profileGamesLoaded;
                    profileGameLoadUs += gameAcquireUs;
                }
            }

            if (game == nullptr) {
                ++casesFailed;
                if (!quiet) {
                    std::cerr << fixture.name << ": failed to load game\n";
                }
                continue;
            }

            TraceFile traceFile;
            try {
                const auto traceParseStart = std::chrono::steady_clock::now();
                traceFile = loadTraceFile(*fixture.traceFile);
                if (profileTimers) {
                    profileTraceParseUs += std::chrono::duration_cast<std::chrono::microseconds>(
                                               std::chrono::steady_clock::now() - traceParseStart)
                                               .count();
                }
            } catch (const std::exception& error) {
                ++casesFailed;
                if (!quiet) {
                    std::cerr << fixture.name << ": " << error.what() << "\n";
                }
                continue;
            }

            for (size_t run = 0; run < repeat; ++run) {
                ps_full_state* session = nullptr;
                ps_error* error = nullptr;
                const auto sessionStart = std::chrono::steady_clock::now();
                if (!ps_full_state_create(game, &session, &error)) {
                    ++casesFailed;
                    if (!quiet) {
                        std::cerr << fixture.name << ": " << ps_error_message(error) << "\n";
                    }
                    ps_free_error(error);
                    if (profileTimers) {
                        profileSessionCreateUs += std::chrono::duration_cast<std::chrono::microseconds>(
                                                      std::chrono::steady_clock::now() - sessionStart)
                                                      .count();
                    }
                    break;
                }
                std::unique_ptr<ps_full_state, decltype(&ps_full_state_destroy)> sessionHolder(session, ps_full_state_destroy);
                ps_full_state_set_unit_testing(session, true);
                if (profileTimers) {
                    profileSessionCreateUs += std::chrono::duration_cast<std::chrono::microseconds>(
                                                  std::chrono::steady_clock::now() - sessionStart)
                                                  .count();
                }

                const auto replayStart = std::chrono::steady_clock::now();
                size_t runSteps = 0;
                const bool ok = replayTraceInputsOnly(session, traceFile, std::cerr, runSteps);
                if (profileTimers) {
                    profileReplayUs += std::chrono::duration_cast<std::chrono::microseconds>(
                                           std::chrono::steady_clock::now() - replayStart)
                                           .count();
                }
                if (!ok) {
                    ++casesFailed;
                    break;
                }
                ++replayRuns;
                replayedSteps += runSteps;
            }
        }
    } catch (const std::exception& error) {
        std::cerr << error.what() << "\n";
        return 1;
    }

    std::cout << "simulation_profile_checked=" << casesChecked << " simulation_profile_failed=" << casesFailed
              << " replay_runs=" << replayRuns << " replay_steps=" << replayedSteps << "\n";

    if (profileTimers) {
        const auto wallUs = std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::steady_clock::now() - wallStart).count();
        const auto usToMs = [](int64_t microseconds) -> int64_t {
            return (microseconds + 500) / 1000;
        };
        std::cerr << "native_simulation_profile test_cases=" << casesChecked << " replay_runs=" << replayRuns
                  << " replay_steps=" << replayedSteps << " wall_ms=" << usToMs(wallUs) << " games_reused=" << profileGamesReused
                  << " games_loaded=" << profileGamesLoaded << " game_reuse_ms=" << usToMs(profileGameReuseUs)
                  << " game_load_ms=" << usToMs(profileGameLoadUs) << " trace_json_parse_ms=" << usToMs(profileTraceParseUs)
                  << " session_create_ms=" << usToMs(profileSessionCreateUs) << " replay_ms=" << usToMs(profileReplayUs) << "\n";
    }

    return casesFailed == 0 ? 0 : 1;
}

int traceAtCommand(const std::string& irPath, const std::string& tracePath, size_t snapshotIndex) {
    ps_game* game = nullptr;
    if (!loadGameFromFile(irPath, &game)) {
        return 1;
    }

    const auto snapshots = loadTraceSnapshots(tracePath);
    if (snapshots.empty()) {
        std::cerr << "Trace has no snapshots\n";
        ps_free_game(game);
        return 1;
    }
    if (snapshotIndex >= snapshots.size()) {
        std::cerr << "Snapshot index out of range: " << snapshotIndex
                  << " >= " << snapshots.size() << "\n";
        ps_free_game(game);
        return 1;
    }

    ps_full_state* session = nullptr;
    ps_error* error = nullptr;
    if (!ps_full_state_create(game, &session, &error)) {
        std::cerr << ps_error_message(error) << "\n";
        ps_free_error(error);
        ps_free_game(game);
        return 1;
    }

    for (size_t index = 1; index <= snapshotIndex; ++index) {
        const auto& snapshot = snapshots[index];
        if (snapshot.phase == "again") {
            (void)ps_full_state_turn(session, PS_INPUT_TICK);
        } else if (snapshot.numericInput.has_value()) {
            (void)ps_full_state_turn(session, static_cast<ps_input>(*snapshot.numericInput));
        } else if (snapshot.stringInput.has_value()) {
            if (*snapshot.stringInput == "tick") {
                (void)ps_full_state_turn(session, PS_INPUT_TICK);
            } else if (*snapshot.stringInput == "restart") {
                if (!ps_full_state_restart(session)) {
                    std::cerr << "Restart failed at snapshot[" << index << "]\n";
                    ps_full_state_destroy(session);
                    ps_free_game(game);
                    return 1;
                }
            } else if (*snapshot.stringInput == "undo") {
                if (!ps_full_state_undo(session)) {
                    std::cerr << "Undo failed at snapshot[" << index << "]\n";
                    ps_full_state_destroy(session);
                    ps_free_game(game);
                    return 1;
                }
            } else {
                std::cerr << "Unsupported trace input token: " << *snapshot.stringInput << "\n";
                ps_full_state_destroy(session);
                ps_free_game(game);
                return 1;
            }
        } else {
            std::cerr << "Snapshot[" << index << "] has no replayable input token\n";
            ps_full_state_destroy(session);
            ps_free_game(game);
            return 1;
        }
    }

    const int result = printSession(session);
    ps_full_state_destroy(session);
    ps_free_game(game);
    return result;
}

int traceStepAtCommand(const std::string& irPath, const std::string& tracePath, size_t snapshotIndex) {
    ps_game* game = nullptr;
    if (!loadGameFromFile(irPath, &game)) {
        return 1;
    }

    const auto snapshots = loadTraceSnapshots(tracePath);
    if (snapshots.empty()) {
        std::cerr << "Trace has no snapshots\n";
        ps_free_game(game);
        return 1;
    }
    if (snapshotIndex >= snapshots.size()) {
        std::cerr << "Snapshot index out of range: " << snapshotIndex
                  << " >= " << snapshots.size() << "\n";
        ps_free_game(game);
        return 1;
    }
    if (snapshotIndex + 1 >= snapshots.size()) {
        std::cerr << "Snapshot index " << snapshotIndex << " has no next snapshot to execute\n";
        ps_free_game(game);
        return 1;
    }

    ps_full_state* session = nullptr;
    ps_error* error = nullptr;
    if (!ps_full_state_create(game, &session, &error)) {
        std::cerr << ps_error_message(error) << "\n";
        ps_free_error(error);
        ps_free_game(game);
        return 1;
    }

    {
        ScopedEnvSilence silence({
            "PS_DEBUG_RANDOM",
            "PS_DEBUG_RANDOM_BOARD_HASH",
            "PS_DEBUG_RANDOM_SESSION_HASH",
            "PS_DEBUG_RANDOM_SUBSTRING",
            "PS_DEBUG_MOVES",
            "PS_DEBUG_RULES",
            "PS_DEBUG_RIGID",
            "PS_DEBUG_AGAIN"
        });
        for (size_t index = 1; index <= snapshotIndex; ++index) {
            const auto& snapshot = snapshots[index];
            if (snapshot.phase == "again") {
                (void)ps_full_state_turn(session, PS_INPUT_TICK);
            } else if (snapshot.numericInput.has_value()) {
                (void)ps_full_state_turn(session, static_cast<ps_input>(*snapshot.numericInput));
            } else if (snapshot.stringInput.has_value()) {
                if (*snapshot.stringInput == "tick") {
                    (void)ps_full_state_turn(session, PS_INPUT_TICK);
                } else if (*snapshot.stringInput == "restart") {
                    if (!ps_full_state_restart(session)) {
                        std::cerr << "Restart failed at snapshot[" << index << "]\n";
                        ps_full_state_destroy(session);
                        ps_free_game(game);
                        return 1;
                    }
                } else if (*snapshot.stringInput == "undo") {
                    if (!ps_full_state_undo(session)) {
                        std::cerr << "Undo failed at snapshot[" << index << "]\n";
                        ps_full_state_destroy(session);
                        ps_free_game(game);
                        return 1;
                    }
                } else {
                    std::cerr << "Unsupported trace input token: " << *snapshot.stringInput << "\n";
                    ps_full_state_destroy(session);
                    ps_free_game(game);
                    return 1;
                }
            } else {
                std::cerr << "Snapshot[" << index << "] has no replayable input token\n";
                ps_full_state_destroy(session);
                ps_free_game(game);
                return 1;
            }
        }
    }

    const auto& nextSnapshot = snapshots[snapshotIndex + 1];
    ps_step_result stepResult{};
    if (nextSnapshot.phase == "again") {
        stepResult = ps_full_state_turn(session, PS_INPUT_TICK);
    } else if (nextSnapshot.numericInput.has_value()) {
        stepResult = ps_full_state_turn(session, static_cast<ps_input>(*nextSnapshot.numericInput));
    } else if (nextSnapshot.stringInput.has_value()) {
        if (*nextSnapshot.stringInput == "tick") {
            stepResult = ps_full_state_turn(session, PS_INPUT_TICK);
        } else if (*nextSnapshot.stringInput == "restart") {
            if (!ps_full_state_restart(session)) {
                std::cerr << "Restart failed at snapshot[" << (snapshotIndex + 1) << "]\n";
                ps_full_state_destroy(session);
                ps_free_game(game);
                return 1;
            }
        } else if (*nextSnapshot.stringInput == "undo") {
            if (!ps_full_state_undo(session)) {
                std::cerr << "Undo failed at snapshot[" << (snapshotIndex + 1) << "]\n";
                ps_full_state_destroy(session);
                ps_free_game(game);
                return 1;
            }
        } else {
            std::cerr << "Unsupported trace input token: " << *nextSnapshot.stringInput << "\n";
            ps_full_state_destroy(session);
            ps_free_game(game);
            return 1;
        }
    } else {
        std::cerr << "Snapshot[" << (snapshotIndex + 1) << "] has no replayable input token\n";
        ps_full_state_destroy(session);
        ps_free_game(game);
        return 1;
    }

    std::cerr << "executed_snapshot=" << (snapshotIndex + 1)
              << " phase=" << nextSnapshot.phase
              << " input=";
    if (nextSnapshot.numericInput.has_value()) {
        std::cerr << *nextSnapshot.numericInput;
    } else if (nextSnapshot.stringInput.has_value()) {
        std::cerr << *nextSnapshot.stringInput;
    } else {
        std::cerr << "null";
    }
    std::cerr << " changed=" << (stepResult.changed ? 1 : 0)
              << " transitioned=" << (stepResult.transitioned ? 1 : 0)
              << " won=" << (stepResult.won ? 1 : 0)
              << " audio_events=" << stepResult.audio_event_count
              << "\n";
    if (stepResult.audio_event_count > 0 && stepResult.audio_events != nullptr) {
        std::cerr << "audio_seeds=";
        for (size_t index = 0; index < stepResult.audio_event_count; ++index) {
            if (index > 0) {
                std::cerr << ",";
            }
            std::cerr << stepResult.audio_events[index].seed;
            if (stepResult.audio_events[index].kind != nullptr && stepResult.audio_events[index].kind[0] != '\0') {
                std::cerr << ":" << stepResult.audio_events[index].kind;
            }
        }
        std::cerr << "\n";
    }

    bool ok = compareSnapshot(nextSnapshot, session, &stepResult, snapshotIndex + 1, std::cerr);
    const int result = printSession(session);
    ps_full_state_destroy(session);
    ps_free_game(game);
    return ok ? result : 1;
}

int runCommand(const std::string& irPath, int argc, char** argv) {
    (void)argc;
    (void)argv;

    ps_game* game = nullptr;
    if (!loadGameFromFile(irPath, &game)) {
        return 1;
    }
    const int result = runCommandForGame(game);
    ps_free_game(game);
    return result;
}

int benchCommand(const std::string& irPath, int argc, char** argv) {
    uint32_t iterations = 10000;
    uint32_t threads = 1;
    for (int index = 0; index < argc; ++index) {
        const std::string arg = argv[index];
        if (arg == "--iterations" && index + 1 < argc) {
            iterations = static_cast<uint32_t>(std::stoul(argv[++index]));
        } else if (arg == "--threads" && index + 1 < argc) {
            threads = static_cast<uint32_t>(std::stoul(argv[++index]));
        }
    }

    ps_game* game = nullptr;
    if (!loadGameFromFile(irPath, &game)) {
        return 1;
    }
    const int result = benchCommandForGame(game, iterations, threads);
    ps_free_game(game);
    return result;
}

int stepCommand(const std::string& irPath, int argc, char** argv) {
    std::vector<std::string> inputTokens;
    for (int index = 0; index < argc; ++index) {
        inputTokens.emplace_back(argv[index]);
    }

    ps_game* game = nullptr;
    if (!loadGameFromFile(irPath, &game)) {
        return 1;
    }
    const int result = stepCommandForGame(game, inputTokens);
    ps_free_game(game);
    return result;
}

void ensureDefaultSourceLoad(std::vector<std::string>& args, bool addSettleAgain = true) {
    bool hasLevel = false;
    bool hasRestart = false;
    bool hasSettleAgain = false;
    for (size_t index = 0; index < args.size(); ++index) {
        if (args[index] == "--level") {
            hasLevel = true;
        } else if (args[index] == "--restart") {
            hasRestart = true;
        } else if (args[index] == "--settle-again") {
            hasSettleAgain = true;
        }
    }
    if (!hasLevel && !hasRestart) {
        args.push_back("--level");
        args.push_back("0");
    }
    if (addSettleAgain && !hasSettleAgain) {
        args.push_back("--settle-again");
    }
}

std::string jsonStringLiteral(std::string_view utf8);

std::string removeAsciiWhitespace(std::string_view value) {
    std::string out;
    out.reserve(value.size());
    for (unsigned char ch : value) {
        if (std::isspace(ch) == 0) {
            out.push_back(static_cast<char>(ch));
        }
    }
    return out;
}

bool expectedDiagnosticsSubsequenceMatch(
    const std::vector<std::string>& expected,
    const std::vector<std::string>& actual
) {
    size_t expectedIndex = 0;
    for (const auto& message : actual) {
        if (expectedIndex >= expected.size()) {
            break;
        }
        if (removeAsciiWhitespace(message) == removeAsciiWhitespace(expected[expectedIndex])) {
            ++expectedIndex;
        }
    }
    return expectedIndex == expected.size();
}

int64_t elapsedMicrosSince(std::chrono::steady_clock::time_point start) {
    return std::chrono::duration_cast<std::chrono::microseconds>(
        std::chrono::steady_clock::now() - start).count();
}

int64_t usToMs(int64_t microseconds) {
    return microseconds / 1000;
}

struct SimulationCorpusOptions {
    size_t progressEvery = 25;
    size_t repeat = 1;
    size_t jobs = 1;
    size_t topSlowCases = 0;
    std::optional<size_t> caseIndex;
    std::optional<std::string> caseNameFilter;
    bool profileTimers = false;
    bool quiet = false;
    bool compactTurnOracle = false;
    bool requireCompactTurnOracleChecks = false;
};

struct SimulationCorpusCase {
    size_t index = 0;
    std::string name;
    std::string source;
    std::vector<std::string> inputs;
    std::string expectedSerialized;
    int32_t targetLevel = 0;
    std::optional<std::string> seed;
};

struct SimulationCaseTiming {
    int64_t sourceCompileUs = 0;
    int64_t sessionCreateUs = 0;
    int64_t levelLoadUs = 0;
    int64_t replayUs = 0;
    int64_t serializeUs = 0;
};

struct SimulationCaseResult {
    bool passed = false;
    std::string error;
    SimulationCaseTiming timing;
    uint64_t compactTurnOracleChecks = 0;
    uint64_t compactTurnOracleHandled = 0;
    uint64_t compactTurnOracleStateChecks = 0;
    uint64_t compactTurnOracleFailures = 0;
};

struct SimulationCompileCache {
    std::vector<std::shared_ptr<ps_game>> games;
    std::vector<std::string> errors;
    std::vector<int64_t> compileUs;
    int64_t totalCompileUs = 0;
    size_t gamesLoaded = 0;
    size_t gamesReused = 0;
};

struct SimulationTimingTotals {
    int64_t testdataParseUs = 0;
    int64_t sourceCompileUs = 0;
    int64_t sessionCreateUs = 0;
    int64_t levelLoadUs = 0;
    int64_t replayUs = 0;
    int64_t serializeUs = 0;
};

SimulationCorpusOptions parseSimulationCorpusOptions(int argc, char** argv) {
    SimulationCorpusOptions options;
    for (int index = 0; index < argc; ++index) {
        const std::string arg = argv[index];
        if (arg == "--progress-every" && index + 1 < argc) {
            options.progressEvery = static_cast<size_t>(std::stoull(argv[++index]));
        } else if (arg == "--repeat" && index + 1 < argc) {
            options.repeat = std::max<size_t>(1, static_cast<size_t>(std::stoull(argv[++index])));
        } else if (arg == "--jobs" && index + 1 < argc) {
            const std::string value = argv[++index];
            if (value == "auto") {
                options.jobs = std::max<size_t>(1, std::thread::hardware_concurrency());
            } else {
                options.jobs = std::max<size_t>(1, static_cast<size_t>(std::stoull(value)));
            }
        } else if (arg == "--profile-timers") {
            options.profileTimers = true;
        } else if (arg == "--top-slow-cases" && index + 1 < argc) {
            options.topSlowCases = static_cast<size_t>(std::stoull(argv[++index]));
        } else if (arg == "--case-index" && index + 1 < argc) {
            options.caseIndex = static_cast<size_t>(std::stoull(argv[++index]));
        } else if (arg == "--case-name" && index + 1 < argc) {
            options.caseNameFilter = argv[++index];
        } else if (arg == "--quiet") {
            options.quiet = true;
            options.progressEvery = 0;
        } else if (arg == "--compact-turn-oracle" || arg == "--compact-tick-oracle") {
            options.compactTurnOracle = true;
        } else if (arg == "--require-compact-turn-oracle-checks" || arg == "--require-compact-tick-oracle-checks") {
            options.requireCompactTurnOracleChecks = true;
        } else {
            throw std::runtime_error("Unsupported simulation-testdata argument: " + arg);
        }
    }
    return options;
}

std::vector<SimulationCorpusCase> filterSimulationCorpusCases(
    const std::vector<SimulationCorpusCase>& cases,
    const SimulationCorpusOptions& options
) {
    if (!options.caseIndex.has_value() && !options.caseNameFilter.has_value()) {
        return cases;
    }

    std::vector<SimulationCorpusCase> filtered;
    if (options.caseIndex.has_value()) {
        const size_t oneBasedIndex = *options.caseIndex;
        if (oneBasedIndex == 0 || oneBasedIndex > cases.size()) {
            throw std::runtime_error("--case-index is out of range");
        }
        filtered.push_back(cases[oneBasedIndex - 1]);
    }
    if (options.caseNameFilter.has_value()) {
        const std::string& needle = *options.caseNameFilter;
        for (const auto& testCase : cases) {
            if (testCase.name.find(needle) != std::string::npos) {
                filtered.push_back(testCase);
            }
        }
    }
    if (filtered.empty()) {
        throw std::runtime_error("No simulation corpus cases matched the requested filter");
    }
    return filtered;
}

std::vector<SimulationCorpusCase> parseSimulationCorpusCases(const puzzlescript::json::Value& root) {
    const auto& rawCases = root.asArray();
    std::vector<SimulationCorpusCase> cases;
    cases.reserve(rawCases.size());
    for (size_t caseIndex = 0; caseIndex < rawCases.size(); ++caseIndex) {
        const auto& entryValue = rawCases[caseIndex];
        if (!entryValue.isArray() || entryValue.asArray().size() < 2) {
            throw std::runtime_error("case[" + std::to_string(caseIndex) + "]: malformed testdata entry");
        }
        const auto& entry = entryValue.asArray();
        if (!entry[1].isArray()) {
            throw std::runtime_error("case[" + std::to_string(caseIndex) + "]: malformed payload");
        }
        const auto& payload = entry[1].asArray();
        if (payload.size() < 3 || !payload[0].isString() || !payload[2].isString()) {
            throw std::runtime_error("case[" + std::to_string(caseIndex) + "]: malformed simulation payload");
        }

        SimulationCorpusCase parsed;
        parsed.index = caseIndex;
        parsed.name = entry[0].isString() ? entry[0].asString() : ("case[" + std::to_string(caseIndex) + "]");
        parsed.source = payload[0].asString();
        if (parsed.source.empty() || parsed.source.back() != '\n') {
            parsed.source.push_back('\n');
        }
        parsed.inputs = payload.size() > 1 ? inputTokensFromJsonArray(payload[1]) : std::vector<std::string>{};
        parsed.expectedSerialized = payload[2].asString();
        parsed.targetLevel = payload.size() >= 4 && !payload[3].isNull() ? looseAsInt(payload[3]) : 0;
        if (payload.size() >= 5 && !payload[4].isNull()) {
            if (payload[4].isString()) {
                parsed.seed = payload[4].asString();
            } else if (payload[4].isInteger() || payload[4].isDouble()) {
                parsed.seed = std::to_string(payload[4].isInteger() ? payload[4].asInteger() : payload[4].asDouble());
            }
        }
        cases.push_back(std::move(parsed));
    }
    return cases;
}

SimulationCompileCache compileSimulationCorpusGames(const std::vector<SimulationCorpusCase>& cases, size_t jobs) {
    SimulationCompileCache cache;
    cache.games.resize(cases.size());
    cache.errors.resize(cases.size());
    cache.compileUs.resize(cases.size(), 0);
    std::unordered_map<std::string, size_t> sourceToUniqueIndex;
    sourceToUniqueIndex.reserve(cases.size());
    std::vector<size_t> caseToUniqueIndex(cases.size(), 0);
    std::vector<std::string> uniqueSources;
    std::vector<std::string> uniqueNames;
    uniqueSources.reserve(cases.size());
    uniqueNames.reserve(cases.size());
    for (size_t caseIndex = 0; caseIndex < cases.size(); ++caseIndex) {
        const SimulationCorpusCase& testCase = cases[caseIndex];
        const auto [it, inserted] = sourceToUniqueIndex.emplace(testCase.source, uniqueSources.size());
        caseToUniqueIndex[caseIndex] = it->second;
        if (inserted) {
            uniqueSources.push_back(testCase.source);
            uniqueNames.push_back(testCase.name);
        }
    }

    std::vector<std::shared_ptr<ps_game>> uniqueGames(uniqueSources.size());
    std::vector<std::string> uniqueErrors(uniqueSources.size());
    std::vector<int64_t> uniqueCompileUs(uniqueSources.size(), 0);
    std::atomic<size_t> nextUnique{0};
    const size_t compileJobs = std::max<size_t>(1, std::min(jobs, std::max<size_t>(uniqueSources.size(), 1)));
    std::vector<std::future<void>> workers;
    workers.reserve(compileJobs);
    for (size_t workerIndex = 0; workerIndex < compileJobs; ++workerIndex) {
        workers.push_back(std::async(std::launch::async, [&]() {
            while (true) {
                const size_t uniqueIndex = nextUnique.fetch_add(1, std::memory_order_relaxed);
                if (uniqueIndex >= uniqueSources.size()) {
                    break;
                }

                const auto phaseStart = std::chrono::steady_clock::now();
                ps_game* rawGame = nullptr;
                if (!loadGameFromSourceText(uniqueSources[uniqueIndex], &rawGame)) {
                    uniqueCompileUs[uniqueIndex] = elapsedMicrosSince(phaseStart);
                    uniqueErrors[uniqueIndex] = uniqueNames[uniqueIndex] + ": failed to compile source";
                    continue;
                }
                uniqueCompileUs[uniqueIndex] = elapsedMicrosSince(phaseStart);
                uniqueGames[uniqueIndex] = std::shared_ptr<ps_game>(rawGame, ps_free_game);
            }
        }));
    }
    for (auto& worker : workers) {
        worker.get();
    }

    for (size_t uniqueIndex = 0; uniqueIndex < uniqueSources.size(); ++uniqueIndex) {
        cache.totalCompileUs += uniqueCompileUs[uniqueIndex];
        if (uniqueGames[uniqueIndex]) {
            ++cache.gamesLoaded;
        }
    }
    cache.gamesReused = cases.size() - uniqueSources.size();
    for (size_t caseIndex = 0; caseIndex < cases.size(); ++caseIndex) {
        const size_t uniqueIndex = caseToUniqueIndex[caseIndex];
        cache.games[caseIndex] = uniqueGames[uniqueIndex];
        cache.errors[caseIndex] = uniqueErrors[uniqueIndex].empty()
            ? std::string{}
            : cases[caseIndex].name + ": failed to compile source";
        cache.compileUs[caseIndex] = uniqueCompileUs[uniqueIndex];
    }
    return cache;
}

SimulationCaseResult runSimulationCorpusCase(
    const SimulationCorpusCase& testCase,
    ps_game* game,
    bool compactTurnOracle
) {
    SimulationCaseResult result;
    if (game == nullptr) {
        result.error = testCase.name + ": failed to compile source";
        return result;
    }

    ps_full_state* rawSession = nullptr;
    ps_error* error = nullptr;
    auto phaseStart = std::chrono::steady_clock::now();
    if (!sessionCreateForGame(game, testCase.seed, &rawSession, &error)) {
        result.timing.sessionCreateUs = elapsedMicrosSince(phaseStart);
        result.error = testCase.name + ": " + ps_error_message(error);
        ps_free_error(error);
        return result;
    }
    result.timing.sessionCreateUs = elapsedMicrosSince(phaseStart);
    std::unique_ptr<ps_full_state, decltype(&ps_full_state_destroy)> session(rawSession, ps_full_state_destroy);
    ps_full_state_set_unit_testing(session.get(), true);

    phaseStart = std::chrono::steady_clock::now();
    if (!ps_full_state_load_level(session.get(), testCase.targetLevel, &error)) {
        result.timing.levelLoadUs = elapsedMicrosSince(phaseStart);
        result.error = testCase.name + ": " + ps_error_message(error);
        ps_free_error(error);
        return result;
    }
    result.timing.levelLoadUs = elapsedMicrosSince(phaseStart);

    phaseStart = std::chrono::steady_clock::now();
    ReplayOracleStats oracleStats;
    ReplayOracleStats* oracleStatsPtr = compactTurnOracle ? &oracleStats : nullptr;
    if (!replayInputTokens(session.get(), testCase.inputs, nullptr, oracleStatsPtr)) {
        result.timing.replayUs = elapsedMicrosSince(phaseStart);
        result.compactTurnOracleChecks = oracleStats.compactTurnChecks;
        result.compactTurnOracleHandled = oracleStats.compactTurnHandled;
        result.compactTurnOracleStateChecks = oracleStats.compactTurnStateChecks;
        result.compactTurnOracleFailures = oracleStats.compactTurnFailures;
        result.error = testCase.name + ": failed to replay inputs";
        if (!oracleStats.firstFailure.empty()) {
            result.error += "\n" + oracleStats.firstFailure;
        }
        return result;
    }
    result.timing.replayUs = elapsedMicrosSince(phaseStart);
    result.compactTurnOracleChecks = oracleStats.compactTurnChecks;
    result.compactTurnOracleHandled = oracleStats.compactTurnHandled;
    result.compactTurnOracleStateChecks = oracleStats.compactTurnStateChecks;
    result.compactTurnOracleFailures = oracleStats.compactTurnFailures;

    phaseStart = std::chrono::steady_clock::now();
    char* serializedRaw = ps_full_state_serialize_test_string(session.get());
    const std::string actualSerialized = serializedRaw ? serializedRaw : "";
    ps_string_free(serializedRaw);
    result.timing.serializeUs = elapsedMicrosSince(phaseStart);

    if (actualSerialized != testCase.expectedSerialized) {
        std::ostringstream stream;
        stream << testCase.name << ": final serialized level mismatch\n"
               << "expected:\n" << testCase.expectedSerialized << "\n"
               << "actual:\n" << actualSerialized << "\n";
        result.error = stream.str();
        return result;
    }

    result.passed = true;
    return result;
}

SimulationTimingTotals sumSimulationTimings(
    const std::vector<SimulationCaseResult>& results,
    int64_t testdataParseUs
) {
    SimulationTimingTotals totals;
    totals.testdataParseUs = testdataParseUs;
    for (const auto& result : results) {
        totals.sourceCompileUs += result.timing.sourceCompileUs;
        totals.sessionCreateUs += result.timing.sessionCreateUs;
        totals.levelLoadUs += result.timing.levelLoadUs;
        totals.replayUs += result.timing.replayUs;
        totals.serializeUs += result.timing.serializeUs;
    }
    return totals;
}

int64_t medianMicros(std::vector<int64_t> values) {
    if (values.empty()) {
        return 0;
    }
    std::sort(values.begin(), values.end());
    return values[values.size() / 2];
}

int runSourceCommand(const std::string& sourcePath, int argc, char** argv) {
    std::vector<std::string> exporterArgs;
    std::vector<std::string> traceArgs;
    bool hasInputTrace = false;
    bool finalOnly = false;
    bool emitJson = false;
    bool nativeCompile = false;
    std::optional<std::string> inputsJson;
    std::optional<std::string> inputsFile;
    std::optional<int32_t> requestedLevel;
    std::optional<std::string> cliLoadedLevelSeed;
    for (int index = 0; index < argc; ++index) {
        const std::string arg = argv[index];
        if (arg == "--headless") {
            continue;
        }
        if (arg == "--native-compile") {
            nativeCompile = true;
            continue;
        }
        if (arg == "--final-only") {
            finalOnly = true;
            continue;
        }
        if (arg == "--json") {
            emitJson = true;
            continue;
        }
        if ((arg == "--level" || arg == "--seed") && index + 1 < argc) {
            const std::string value = argv[++index];
            exporterArgs.push_back(arg);
            exporterArgs.push_back(value);
            traceArgs.push_back(arg);
            traceArgs.push_back(value);
            if (arg == "--level") {
                requestedLevel = static_cast<int32_t>(std::stoi(value));
            } else {
                cliLoadedLevelSeed = value;
            }
            continue;
        }
        if (arg == "--settle-again") {
            exporterArgs.push_back(arg);
            continue;
        }
        if ((arg == "--inputs-json" || arg == "--inputs-file") && index + 1 < argc) {
            hasInputTrace = true;
            const std::string value = argv[++index];
            if (arg == "--inputs-json") {
                inputsJson = value;
            } else {
                inputsFile = value;
            }
            traceArgs.push_back(arg);
            traceArgs.push_back(value);
            continue;
        }
        exporterArgs.emplace_back(arg);
    }
    ensureDefaultSourceLoad(exporterArgs);
    ensureDefaultSourceLoad(traceArgs, false);
    if (!requestedLevel.has_value()) {
        for (size_t i = 0; i + 1 < exporterArgs.size(); ++i) {
            if (exporterArgs[i] == "--level") {
                requestedLevel = static_cast<int32_t>(std::stoi(exporterArgs[i + 1]));
                break;
            }
        }
    }

    ps_game* game = nullptr;
    if (!nativeCompile) {
        if (!loadGameFromJsonText(runIrExporterAndCaptureJson(sourcePath, exporterArgs), &game)) {
            return 1;
        }
    } else {
        if (!loadGameFromSourceFile(sourcePath, &game)) {
            return 1;
        }
    }
    int result = 0;
    if (!hasInputTrace) {
        result = runCommandForGame(game);
    } else if (!finalOnly) {
        const std::optional<std::string> sessionSeed = nativeCompile ? cliLoadedLevelSeed : std::nullopt;
        const std::optional<int32_t> traceLevel = nativeCompile ? requestedLevel : std::optional<int32_t>{};
        result = diffTraceAgainstSnapshots(
            game,
            loadTraceSnapshotsFromJsonText(runTraceExporterAndCaptureJson(sourcePath, traceArgs)),
            std::cerr,
            true,
            sessionSeed,
            traceLevel
        );
    } else {
        ps_full_state* session = nullptr;
        ps_error* error = nullptr;
        const std::optional<std::string> sessionSeed = nativeCompile ? cliLoadedLevelSeed : std::nullopt;
        if (!sessionCreateForGame(game, sessionSeed, &session, &error)) {
            std::cerr << ps_error_message(error) << "\n";
            ps_free_error(error);
            ps_free_game(game);
            return 1;
        }
        if (nativeCompile && requestedLevel.has_value()) {
            if (!ps_full_state_load_level(session, *requestedLevel, &error)) {
                std::cerr << ps_error_message(error) << "\n";
                ps_free_error(error);
                ps_full_state_destroy(session);
                ps_free_game(game);
                return 1;
            }
        }

        std::vector<std::string> tokens;
        if (inputsJson.has_value()) {
            tokens = loadInputTokensFromJsonText(*inputsJson);
        } else if (inputsFile.has_value()) {
            tokens = loadInputTokensFromJsonFile(*inputsFile);
        } else {
            tokens.clear();
        }

        std::vector<std::string> sounds;
        sounds.reserve(16);
        if (!replayInputTokens(session, tokens, emitJson ? &sounds : nullptr)) {
            ps_full_state_destroy(session);
            ps_free_game(game);
            return 1;
        }

        char* serialized = ps_full_state_serialize_test_string(session);
        const std::string actualFinal = serialized ? serialized : "";
        ps_string_free(serialized);

        if (!emitJson) {
            std::cout << actualFinal;
            if (!actualFinal.empty() && actualFinal.back() != '\n') {
                std::cout << "\n";
            }
        } else {
            std::cout << "{"
                      << "\"serialized_level\":" << jsonStringLiteral(actualFinal)
                      << ",\"sounds\":[";
            for (size_t i = 0; i < sounds.size(); ++i) {
                if (i > 0) {
                    std::cout << ",";
                }
                std::cout << jsonStringLiteral(sounds[i]);
            }
            std::cout << "]"
                      << "}\n";
        }

        ps_full_state_destroy(session);
        result = 0;
    }
    ps_free_game(game);
    return result;
}

int simulationTestdataCommand(const std::filesystem::path& testdataPath, int argc, char** argv) {
    const SimulationCorpusOptions options = parseSimulationCorpusOptions(argc, argv);
    const auto wallStartedAt = std::chrono::steady_clock::now();
    const auto parseStartedAt = std::chrono::steady_clock::now();
    const auto root = loadJsDataArrayAsJson(testdataPath);
    const std::vector<SimulationCorpusCase> allCases = parseSimulationCorpusCases(root);
    const std::vector<SimulationCorpusCase> cases = filterSimulationCorpusCases(allCases, options);
    const int64_t testdataParseUs = elapsedMicrosSince(parseStartedAt);
    const SimulationCompileCache compileCache = compileSimulationCorpusGames(cases, options.jobs);
    const size_t totalChecks = cases.size() * options.repeat;
    std::vector<SimulationCaseResult> results(totalChecks);
    size_t passed = 0;
    size_t failed = 0;

    if (options.profileTimers) {
        ps_runtime_counters_reset();
        ps_runtime_counters_set_enabled(true);
    }

    if (options.jobs <= 1 || totalChecks <= 1) {
        for (size_t checkIndex = 0; checkIndex < totalChecks; ++checkIndex) {
            const size_t caseIndex = checkIndex % cases.size();
            const SimulationCorpusCase& testCase = cases[caseIndex];
            if (!compileCache.errors[caseIndex].empty()) {
                results[checkIndex].error = compileCache.errors[caseIndex];
            } else {
                results[checkIndex] = runSimulationCorpusCase(
                    testCase,
                    compileCache.games[caseIndex].get(),
                    options.compactTurnOracle
                );
            }
            if (results[checkIndex].passed) {
                ++passed;
            } else {
                ++failed;
                if (!results[checkIndex].error.empty()) {
                    std::cerr << results[checkIndex].error;
                    if (results[checkIndex].error.back() != '\n') {
                        std::cerr << "\n";
                    }
                }
            }
            if (!options.quiet && options.progressEvery > 0 && ((checkIndex + 1) % options.progressEvery) == 0) {
                std::cerr << "progress checks=" << (checkIndex + 1) << "/" << totalChecks
                          << " passed=" << passed << " failed=" << failed << "\n";
            }
        }
    } else {
        std::atomic<size_t> nextCheck{0};
        std::vector<std::future<void>> workers;
        workers.reserve(options.jobs);
        for (size_t workerIndex = 0; workerIndex < options.jobs; ++workerIndex) {
            workers.push_back(std::async(std::launch::async, [&]() {
                while (true) {
                    const size_t checkIndex = nextCheck.fetch_add(1, std::memory_order_relaxed);
                    if (checkIndex >= totalChecks) {
                        break;
                    }
                    const size_t caseIndex = checkIndex % cases.size();
                    const SimulationCorpusCase& testCase = cases[caseIndex];
                    if (!compileCache.errors[caseIndex].empty()) {
                        results[checkIndex].error = compileCache.errors[caseIndex];
                    } else {
                        results[checkIndex] = runSimulationCorpusCase(
                            testCase,
                            compileCache.games[caseIndex].get(),
                            options.compactTurnOracle
                        );
                    }
                }
            }));
        }
        for (auto& worker : workers) {
            worker.get();
        }
        for (size_t checkIndex = 0; checkIndex < totalChecks; ++checkIndex) {
            if (results[checkIndex].passed) {
                ++passed;
            } else {
                ++failed;
                if (!results[checkIndex].error.empty()) {
                    std::cerr << results[checkIndex].error;
                    if (results[checkIndex].error.back() != '\n') {
                        std::cerr << "\n";
                    }
                }
            }
        }
    }

    ps_runtime_counters counters{};
    if (options.profileTimers) {
        ps_runtime_counters_snapshot(&counters);
        ps_runtime_counters_set_enabled(false);
    }

    const int64_t wallUs = elapsedMicrosSince(wallStartedAt);
    const SimulationTimingTotals timings = sumSimulationTimings(results, testdataParseUs);
    SimulationTimingTotals reportedTimings = timings;
    reportedTimings.sourceCompileUs += compileCache.totalCompileUs;
    uint64_t compactTurnOracleChecks = 0;
    uint64_t compactTurnOracleHandled = 0;
    uint64_t compactTurnOracleStateChecks = 0;
    uint64_t compactTurnOracleFailures = 0;
    for (const auto& result : results) {
        compactTurnOracleChecks += result.compactTurnOracleChecks;
        compactTurnOracleHandled += result.compactTurnOracleHandled;
        compactTurnOracleStateChecks += result.compactTurnOracleStateChecks;
        compactTurnOracleFailures += result.compactTurnOracleFailures;
    }
    if (options.requireCompactTurnOracleChecks && compactTurnOracleChecks == 0) {
        ++failed;
        std::cerr << "compact turn oracle checks were required but none ran\n";
    }
    std::vector<int64_t> replayUsByRepeat;
    std::vector<int64_t> sourceCompileUsByRepeat;
    if (options.repeat > 0 && !cases.empty()) {
        replayUsByRepeat.assign(options.repeat, 0);
        sourceCompileUsByRepeat.assign(options.repeat, 0);
        const int64_t amortizedSourceCompileUs = compileCache.totalCompileUs
            / static_cast<int64_t>(std::max<size_t>(options.repeat, 1));
        std::fill(sourceCompileUsByRepeat.begin(), sourceCompileUsByRepeat.end(), amortizedSourceCompileUs);
        for (size_t checkIndex = 0; checkIndex < totalChecks; ++checkIndex) {
            const size_t repeatIndex = checkIndex / cases.size();
            if (repeatIndex >= options.repeat) {
                continue;
            }
            replayUsByRepeat[repeatIndex] += results[checkIndex].timing.replayUs;
        }
    }
    std::cout << "cpp_simulation_tests_direct passed=" << passed << " failed=" << failed
              << " total=" << totalChecks << " cases=" << cases.size()
              << " repeats=" << options.repeat << " jobs=" << options.jobs
              << " elapsed_ms=" << usToMs(wallUs);
    if (options.compactTurnOracle || compactTurnOracleChecks > 0 || compactTurnOracleFailures > 0) {
        std::cout << " compact_turn_oracle_checks=" << compactTurnOracleChecks
                  << " compact_turn_oracle_handled=" << compactTurnOracleHandled
                  << " compact_turn_oracle_state_checks=" << compactTurnOracleStateChecks
                  << " compact_turn_oracle_failures=" << compactTurnOracleFailures;
    }
    std::cout << "\n";
    if (options.profileTimers) {
        std::cout << "cpp_simulation_profile"
                  << " cases=" << cases.size()
                  << " repeats=" << options.repeat
                  << " checks=" << totalChecks
                  << " jobs=" << options.jobs
                  << " wall_ms=" << usToMs(wallUs)
                  << " games_loaded=" << compileCache.gamesLoaded
                  << " games_reused=" << compileCache.gamesReused
                  << " testdata_parse_ms=" << usToMs(reportedTimings.testdataParseUs)
                  << " source_compile_ms=" << usToMs(reportedTimings.sourceCompileUs)
                  << " session_create_ms=" << usToMs(reportedTimings.sessionCreateUs)
                  << " level_load_ms=" << usToMs(reportedTimings.levelLoadUs)
                  << " replay_ms=" << usToMs(reportedTimings.replayUs)
                  << " replay_avg_ms=" << usToMs(reportedTimings.replayUs / static_cast<int64_t>(std::max<size_t>(options.repeat, 1)))
                  << " replay_median_ms=" << usToMs(medianMicros(replayUsByRepeat))
                  << " serialize_ms=" << usToMs(reportedTimings.serializeUs)
                  << " source_compile_avg_ms=" << usToMs(reportedTimings.sourceCompileUs / static_cast<int64_t>(std::max<size_t>(options.repeat, 1)))
                  << " source_compile_median_ms=" << usToMs(medianMicros(sourceCompileUsByRepeat))
                  << " rules_visited=" << counters.rules_visited
                  << " rules_skipped_by_mask=" << counters.rules_skipped_by_mask
                  << " candidate_cells_tested=" << counters.candidate_cells_tested
                  << " pattern_tests=" << counters.pattern_tests
                  << " pattern_matches=" << counters.pattern_matches
                  << " replacements_attempted=" << counters.replacements_attempted
                  << " replacements_applied=" << counters.replacements_applied
                  << " row_scans=" << counters.row_scans
                  << " ellipsis_scans=" << counters.ellipsis_scans
                  << " mask_rebuild_calls=" << counters.mask_rebuild_calls
                  << " mask_rebuild_dirty_calls=" << counters.mask_rebuild_dirty_calls
                  << " mask_rebuild_rows=" << counters.mask_rebuild_rows
                  << " mask_rebuild_columns=" << counters.mask_rebuild_columns
                  << " specialized_rulegroup_attempts=" << counters.specialized_rulegroup_attempts
                  << " specialized_rulegroup_hits=" << counters.specialized_rulegroup_hits
                  << " specialized_rulegroup_fallbacks=" << counters.specialized_rulegroup_fallbacks
                  << " specialized_full_turn_attempts=" << counters.specialized_full_turn_attempts
                  << " specialized_full_turn_hits=" << counters.specialized_full_turn_hits
                  << " specialized_full_turn_fallbacks=" << counters.specialized_full_turn_fallbacks
                  << " compact_turn_native_calls=" << counters.compact_turn_native_calls
                  << " compact_turn_bridge_calls=" << counters.compact_turn_bridge_calls
                  << " compact_turn_setup_ns=" << counters.compact_turn_setup_ns
                  << " compact_turn_early_rules_ns=" << counters.compact_turn_early_rules_ns
                  << " compact_turn_movement_ns=" << counters.compact_turn_movement_ns
                  << " compact_turn_late_rules_ns=" << counters.compact_turn_late_rules_ns
                  << " compact_turn_win_ns=" << counters.compact_turn_win_ns
                  << " compact_turn_canonicalize_ns=" << counters.compact_turn_canonicalize_ns
                  << " compact_turn_bridge_create_ns=" << counters.compact_turn_bridge_create_ns
                  << " compact_turn_bridge_materialize_ns=" << counters.compact_turn_bridge_materialize_ns
                  << " compact_turn_bridge_turn_ns=" << counters.compact_turn_bridge_turn_ns
                  << " compact_turn_bridge_copyback_ns=" << counters.compact_turn_bridge_copyback_ns
                  << "\n";
    }
    if (options.topSlowCases > 0 && !cases.empty()) {
        struct SlowCase {
            size_t caseIndex = 0;
            int64_t totalUs = 0;
            int64_t sourceCompileUs = 0;
            int64_t replayUs = 0;
            int64_t sessionCreateUs = 0;
            int64_t levelLoadUs = 0;
            int64_t serializeUs = 0;
        };
        std::vector<SlowCase> slowCases(cases.size());
        for (size_t caseIndex = 0; caseIndex < cases.size(); ++caseIndex) {
            slowCases[caseIndex].caseIndex = caseIndex;
            slowCases[caseIndex].sourceCompileUs = compileCache.compileUs[caseIndex];
            slowCases[caseIndex].totalUs += compileCache.compileUs[caseIndex];
        }
        for (size_t checkIndex = 0; checkIndex < results.size(); ++checkIndex) {
            const size_t caseIndex = checkIndex % cases.size();
            const auto& timing = results[checkIndex].timing;
            auto& slow = slowCases[caseIndex];
            slow.sessionCreateUs += timing.sessionCreateUs;
            slow.levelLoadUs += timing.levelLoadUs;
            slow.replayUs += timing.replayUs;
            slow.serializeUs += timing.serializeUs;
            slow.totalUs += timing.sessionCreateUs + timing.levelLoadUs + timing.replayUs + timing.serializeUs;
        }
        std::sort(slowCases.begin(), slowCases.end(), [](const SlowCase& lhs, const SlowCase& rhs) {
            if (lhs.totalUs != rhs.totalUs) {
                return lhs.totalUs > rhs.totalUs;
            }
            return lhs.caseIndex < rhs.caseIndex;
        });
        const size_t emitCount = std::min(options.topSlowCases, slowCases.size());
        for (size_t rank = 0; rank < emitCount; ++rank) {
            const auto& slow = slowCases[rank];
            const auto& testCase = cases[slow.caseIndex];
            std::cout << "cpp_simulation_slow_case"
                      << " rank=" << (rank + 1)
                      << " index=" << (testCase.index + 1)
                      << " name=" << jsonStringLiteral(testCase.name)
                      << " total_ms=" << usToMs(slow.totalUs)
                      << " source_compile_ms=" << usToMs(slow.sourceCompileUs)
                      << " replay_ms=" << usToMs(slow.replayUs)
                      << " session_create_ms=" << usToMs(slow.sessionCreateUs)
                      << " level_load_ms=" << usToMs(slow.levelLoadUs)
                      << " serialize_ms=" << usToMs(slow.serializeUs)
                      << "\n";
        }
    }
    return failed == 0 ? 0 : 1;
}

int compilationTestdataCommand(const std::filesystem::path& testdataPath, int argc, char** argv) {
    size_t progressEvery = 50;
    for (int index = 0; index < argc; ++index) {
        const std::string arg = argv[index];
        if (arg == "--progress-every" && index + 1 < argc) {
            progressEvery = static_cast<size_t>(std::stoull(argv[++index]));
        } else {
            throw std::runtime_error("Unsupported compilation-testdata argument: " + arg);
        }
    }

    const auto startedAt = std::chrono::steady_clock::now();
    const auto root = loadJsDataArrayAsJson(testdataPath);
    const auto& cases = root.asArray();
    size_t passed = 0;
    size_t failed = 0;

    for (size_t caseIndex = 0; caseIndex < cases.size(); ++caseIndex) {
        const auto& entryValue = cases[caseIndex];
        if (!entryValue.isArray() || entryValue.asArray().size() < 2) {
            std::cerr << "case[" << caseIndex << "]: malformed errormessage entry\n";
            ++failed;
            continue;
        }
        const auto& entry = entryValue.asArray();
        const std::string name = entry[0].isString() ? entry[0].asString() : ("case[" + std::to_string(caseIndex) + "]");
        if (!entry[1].isArray()) {
            std::cerr << name << ": malformed payload\n";
            ++failed;
            continue;
        }
        const auto& payload = entry[1].asArray();
        if (payload.size() < 2 || !payload[0].isString() || !payload[1].isArray()) {
            std::cerr << name << ": malformed diagnostics payload\n";
            ++failed;
            continue;
        }

        std::string source = payload[0].asString();
        if (source.empty() || source.back() != '\n') {
            source.push_back('\n');
        }
        std::vector<std::string> expected;
        for (const auto& item : payload[1].asArray()) {
            if (!item.isString()) {
                throw std::runtime_error("diagnostics payload contains non-string expected message: " + name);
            }
            expected.push_back(item.asString());
        }
        const size_t expectedCount = payload.size() >= 3 && (payload[2].isInteger() || payload[2].isDouble())
            ? static_cast<size_t>(looseAsInt(payload[2]))
            : expected.size();

        std::unique_ptr<ps_compiler_result, decltype(&ps_compiler_result_free)> result(
            ps_compiler_compile_source_diagnostics(source.data(), source.size()),
            ps_compiler_result_free
        );
        if (!result) {
            std::cerr << name << ": failed to parse source\n";
            ++failed;
            continue;
        }

        std::vector<std::string> actual;
        size_t actualErrorCount = 0;
        const size_t diagnosticCount = ps_compiler_result_diagnostic_count(result.get());
        actual.reserve(diagnosticCount);
        for (size_t diagnosticIndex = 0; diagnosticIndex < diagnosticCount; ++diagnosticIndex) {
            const ps_diagnostic* diagnostic = ps_compiler_result_diagnostic(result.get(), diagnosticIndex);
            if (diagnostic != nullptr && diagnostic->message != nullptr) {
                if (diagnostic->severity == PS_DIAG_ERROR) {
                    ++actualErrorCount;
                }
                actual.emplace_back(diagnostic->message);
            }
        }

        const bool countOk = actualErrorCount == expectedCount;
        const bool messagesOk = expectedDiagnosticsSubsequenceMatch(expected, actual);
        if (!countOk || !messagesOk) {
            std::cerr << name << ": diagnostics mismatch\n";
            std::cerr << "expected_count=" << expectedCount << " actual_count=" << actualErrorCount << "\n";
            if (!messagesOk) {
                std::cerr << "expected_messages=[";
                for (size_t index = 0; index < expected.size(); ++index) {
                    if (index > 0) std::cerr << ",";
                    std::cerr << jsonStringLiteral(expected[index]);
                }
                std::cerr << "]\nactual_messages=[";
                for (size_t index = 0; index < actual.size(); ++index) {
                    if (index > 0) std::cerr << ",";
                    std::cerr << jsonStringLiteral(actual[index]);
                }
                std::cerr << "]\n";
            }
            ++failed;
            continue;
        }

        ++passed;
        if (progressEvery > 0 && ((caseIndex + 1) % progressEvery) == 0) {
            std::cerr << "progress cases=" << (caseIndex + 1) << "/" << cases.size()
                      << " passed=" << passed << " failed=" << failed << "\n";
        }
    }

    const auto elapsedMs = std::chrono::duration_cast<std::chrono::milliseconds>(
        std::chrono::steady_clock::now() - startedAt).count();
    std::cout << "cpp_compilation_tests_direct passed=" << passed << " failed=" << failed
              << " total=" << cases.size() << " elapsed_ms=" << elapsedMs << "\n";
    return failed == 0 ? 0 : 1;
}

int benchSourceCommand(const std::string& sourcePath, int argc, char** argv) {
    uint32_t iterations = 10000;
    uint32_t threads = 1;
    std::vector<std::string> exporterArgs;
    bool nativeCompile = false;

    for (int index = 0; index < argc; ++index) {
        const std::string arg = argv[index];
        if (arg == "--iterations" && index + 1 < argc) {
            iterations = static_cast<uint32_t>(std::stoul(argv[++index]));
        } else if (arg == "--threads" && index + 1 < argc) {
            threads = static_cast<uint32_t>(std::stoul(argv[++index]));
        } else if (arg == "--native-compile") {
            nativeCompile = true;
        } else {
            exporterArgs.emplace_back(arg);
        }
    }
    ensureDefaultSourceLoad(exporterArgs);

    ps_game* game = nullptr;
    if (!nativeCompile) {
        if (!loadGameFromJsonText(runIrExporterAndCaptureJson(sourcePath, exporterArgs), &game)) {
            return 1;
        }
    } else {
        if (!loadGameFromSourceFile(sourcePath, &game)) {
            return 1;
        }
    }
    const std::optional<std::string> sessionSeed = nativeCompile ? findArgValue(exporterArgs, "--seed") : std::nullopt;
    const int result = benchCommandForGame(game, iterations, threads, sessionSeed);
    ps_free_game(game);
    return result;
}

int playSourceCommand(const std::string& sourcePath, int argc, char** argv) {
    for (int index = 0; index < argc; ++index) {
        const std::string arg = argv[index];
        if (arg == "--native-compile") {
            // Kept temporarily as a no-op for old shell history; source play is
            // native by default now.
            continue;
        }
        throw std::runtime_error("Unsupported play argument: " + arg + "\nTry: puzzlescript_cpp help play");
    }

    ps_game* game = nullptr;
    if (!loadGameFromSourceFile(sourcePath, &game)) {
        return 1;
    }
#ifdef PS_HAVE_SDL2
    const int result = puzzlescript_cpp_run_player_for_game(game, sourcePath);
    ps_free_game(game);
    return result;
#else
    ps_free_game(game);
    std::cerr << "SDL2 support is not enabled in this build.\n";
    return 1;
#endif
}

std::string jsonStringLiteral(std::string_view utf8) {
    std::string out;
    out.reserve(utf8.size() + 2);
    out.push_back('"');
    for (unsigned char ch : utf8) {
        switch (ch) {
            case '"':
                out += "\\\"";
                break;
            case '\\':
                out += "\\\\";
                break;
            case '\b':
                out += "\\b";
                break;
            case '\f':
                out += "\\f";
                break;
            case '\n':
                out += "\\n";
                break;
            case '\r':
                out += "\\r";
                break;
            case '\t':
                out += "\\t";
                break;
            default:
                if (ch < 0x20U) {
                    static const char hex[] = "0123456789abcdef";
                    out += "\\u00";
                    out.push_back(hex[(ch >> 4U) & 0x0FU]);
                    out.push_back(hex[ch & 0x0FU]);
                } else {
                    out.push_back(static_cast<char>(ch));
                }
                break;
        }
    }
    out.push_back('"');
    return out;
}

void appendJsonStringArray(std::ostream& out, const std::vector<std::string>& values) {
    out << "[";
    for (size_t index = 0; index < values.size(); ++index) {
        if (index != 0) out << ",";
        out << jsonStringLiteral(values[index]);
    }
    out << "]";
}

void appendJsonIntArray(std::ostream& out, const std::vector<int32_t>& values) {
    out << "[";
    for (size_t index = 0; index < values.size(); ++index) {
        if (index != 0) out << ",";
        out << values[index];
    }
    out << "]";
}

#if PS_MASK_WORD_BITS == 64
void appendJsonIntArray(std::ostream& out, const puzzlescript::MaskVector& values) {
    out << "[";
    for (size_t index = 0; index < values.size(); ++index) {
        if (index != 0) out << ",";
        out << values[index];
    }
    out << "]";
}
#endif

void appendJsonMask(std::ostream& out, const puzzlescript::Game& game, puzzlescript::MaskOffset offset, uint32_t width) {
    out << "[";
    for (uint32_t index = 0; index < width; ++index) {
        if (index != 0) out << ",";
        if (offset == puzzlescript::kNullMaskOffset || static_cast<size_t>(offset + index) >= game.maskArena.size()) {
            out << 0;
        } else {
            out << game.maskArena[static_cast<size_t>(offset + index)];
        }
    }
    out << "]";
}

bool maskHasAnyBit(const puzzlescript::Game& game, puzzlescript::MaskOffset offset, uint32_t width) {
    if (offset == puzzlescript::kNullMaskOffset) {
        return false;
    }
    for (uint32_t index = 0; index < width; ++index) {
        if (static_cast<size_t>(offset + index) < game.maskArena.size()
            && game.maskArena[static_cast<size_t>(offset + index)] != 0) {
            return true;
        }
    }
    return false;
}

bool maskBitIsSet(const puzzlescript::Game& game, puzzlescript::MaskOffset offset, uint32_t logicalBitIndex) {
    if (offset == puzzlescript::kNullMaskOffset) {
        return false;
    }
    const uint32_t wordIndex = logicalBitIndex / puzzlescript::kMaskWordBits;
    const uint32_t bitIndex = logicalBitIndex % puzzlescript::kMaskWordBits;
    const size_t arenaIndex = static_cast<size_t>(offset + wordIndex);
    if (arenaIndex >= game.maskArena.size()) {
        return false;
    }
    const auto word = static_cast<puzzlescript::MaskWordUnsigned>(game.maskArena[arenaIndex]);
    return (word & (puzzlescript::MaskWordUnsigned{1} << bitIndex)) != 0;
}

bool maskSubsetOf(
    const puzzlescript::Game& game,
    puzzlescript::MaskOffset left,
    puzzlescript::MaskOffset right,
    uint32_t width
) {
    for (uint32_t index = 0; index < width; ++index) {
        const auto leftWord = (left == puzzlescript::kNullMaskOffset || static_cast<size_t>(left + index) >= game.maskArena.size())
            ? puzzlescript::MaskWord{0}
            : game.maskArena[static_cast<size_t>(left + index)];
        const auto rightWord = (right == puzzlescript::kNullMaskOffset || static_cast<size_t>(right + index) >= game.maskArena.size())
            ? puzzlescript::MaskWord{0}
            : game.maskArena[static_cast<size_t>(right + index)];
        if ((leftWord & ~rightWord) != 0) {
            return false;
        }
    }
    return true;
}

bool masksOverlap(
    const puzzlescript::Game& game,
    puzzlescript::MaskOffset left,
    puzzlescript::MaskOffset right,
    uint32_t width
) {
    for (uint32_t index = 0; index < width; ++index) {
        const auto leftWord = (left == puzzlescript::kNullMaskOffset || static_cast<size_t>(left + index) >= game.maskArena.size())
            ? puzzlescript::MaskWord{0}
            : game.maskArena[static_cast<size_t>(left + index)];
        const auto rightWord = (right == puzzlescript::kNullMaskOffset || static_cast<size_t>(right + index) >= game.maskArena.size())
            ? puzzlescript::MaskWord{0}
            : game.maskArena[static_cast<size_t>(right + index)];
        if ((leftWord & rightWord) != 0) {
            return true;
        }
    }
    return false;
}

bool patternImpossible(const puzzlescript::Game& game, const puzzlescript::Pattern& pattern) {
    if (pattern.kind == puzzlescript::Pattern::Kind::Ellipsis) {
        return false;
    }
    if (masksOverlap(game, pattern.objectsPresent, pattern.objectsMissing, game.wordCount)
        || masksOverlap(game, pattern.movementsPresent, pattern.movementsMissing, game.movementWordCount)) {
        return true;
    }
    std::vector<uint8_t> requiredLayers(static_cast<size_t>(game.layerCount), 0);
    for (uint32_t objectId = 0; objectId < game.objectCount; ++objectId) {
        if (!maskBitIsSet(game, pattern.objectsPresent, objectId)) {
            continue;
        }
        const int32_t layer = game.objectsById[static_cast<size_t>(objectId)].layer;
        if (layer < 0 || layer >= game.layerCount) {
            continue;
        }
        auto& seen = requiredLayers[static_cast<size_t>(layer)];
        if (seen != 0) {
            return true;
        }
        seen = 1;
    }
    return false;
}

bool ruleImpossible(const puzzlescript::Game& game, const puzzlescript::Rule& rule) {
    for (const auto& row : rule.patterns) {
        for (const auto& pattern : row) {
            if (patternImpossible(game, pattern)) {
                return true;
            }
        }
    }
    return false;
}

bool replacementGuaranteedNoop(
    const puzzlescript::Game& game,
    const puzzlescript::Replacement& repl,
    const puzzlescript::Pattern& pattern
) {
    if (maskHasAnyBit(game, repl.objectsSet, game.wordCount)
        || maskHasAnyBit(game, repl.movementsSet, game.movementWordCount)
        || repl.hasMovementsLayerMask
        || repl.hasRandomEntityMask
        || repl.hasRandomDirMask) {
        return false;
    }
    return maskSubsetOf(game, repl.objectsClear, pattern.objectsMissing, game.wordCount)
        && maskSubsetOf(game, repl.movementsClear, pattern.movementsMissing, game.movementWordCount);
}

std::pair<int32_t, int32_t> ruleDirectionDelta(int32_t directionMask) {
    switch (directionMask) {
        case 1: return {0, -1};
        case 2: return {0, 1};
        case 4: return {-1, 0};
        case 8: return {1, 0};
        default: return {0, 0};
    }
}

void appendJsonIntArrayUnique(std::ostream& out, std::vector<int32_t> values) {
    std::sort(values.begin(), values.end());
    values.erase(std::unique(values.begin(), values.end()), values.end());
    appendJsonIntArray(out, values);
}

void appendObjectIdsFromMask(std::ostream& out, const puzzlescript::Game& game, puzzlescript::MaskOffset offset) {
    out << "[";
    bool first = true;
    for (uint32_t objectId = 0; objectId < game.objectCount; ++objectId) {
        if (!maskBitIsSet(game, offset, objectId)) {
            continue;
        }
        if (!first) {
            out << ",";
        }
        first = false;
        out << objectId;
    }
    out << "]";
}

void appendMovementBitPairs(std::ostream& out, const puzzlescript::Game& game, puzzlescript::MaskOffset offset) {
    out << "[";
    bool first = true;
    for (uint32_t layerIndex = 0; layerIndex < game.layerCount; ++layerIndex) {
        for (uint32_t movementBit = 0; movementBit < 5; ++movementBit) {
            if (!maskBitIsSet(game, offset, layerIndex * 5U + movementBit)) {
                continue;
            }
            if (!first) {
                out << ",";
            }
            first = false;
            out << "[" << layerIndex << "," << movementBit << "]";
        }
    }
    out << "]";
}

void appendMovementLayersFromMask(std::ostream& out, const puzzlescript::Game& game, puzzlescript::MaskOffset offset) {
    out << "[";
    bool first = true;
    for (uint32_t layerIndex = 0; layerIndex < game.layerCount; ++layerIndex) {
        bool hasLayerBit = false;
        for (uint32_t movementBit = 0; movementBit < 5; ++movementBit) {
            if (maskBitIsSet(game, offset, layerIndex * 5U + movementBit)) {
                hasLayerBit = true;
                break;
            }
        }
        if (!hasLayerBit) {
            continue;
        }
        if (!first) {
            out << ",";
        }
        first = false;
        out << layerIndex;
    }
    out << "]";
}

void appendRulePlanJson(std::ostream& out, const puzzlescript::Game& game) {
    auto appendRuleEntry = [&](const puzzlescript::Rule& rule, size_t groupIndex, size_t ruleIndex, bool late) {
        const auto [dx, dy] = ruleDirectionDelta(rule.direction);
        bool hasEllipsis = false;
        for (const auto count : rule.ellipsisCount) {
            if (count > 0) {
                hasEllipsis = true;
                break;
            }
        }
        std::vector<std::string> commandNames;
        commandNames.reserve(rule.commands.size());
        for (const auto& command : rule.commands) {
            commandNames.push_back(command.name);
        }
        const bool simpleDeterministicRowRule =
            !rule.isRandom
            && !rule.rigid
            && !hasEllipsis
            && rule.patterns.size() == 1
            && commandNames.empty();
        out << "{";
        out << "\"rule_index\":" << ruleIndex
            << ",\"group_index\":" << groupIndex
            << ",\"late\":" << (late ? "true" : "false")
            << ",\"direction\":" << rule.direction
            << ",\"line_number\":" << rule.lineNumber
            << ",\"group_number\":" << rule.groupNumber
            << ",\"rigid\":" << (rule.rigid ? "true" : "false")
            << ",\"is_random\":" << (rule.isRandom ? "true" : "false")
            << ",\"has_replacements\":" << (rule.hasReplacements ? "true" : "false")
            << ",\"has_ellipsis\":" << (hasEllipsis ? "true" : "false")
            << ",\"row_count\":" << rule.patterns.size()
            << ",\"has_commands\":" << (!commandNames.empty() ? "true" : "false")
            << ",\"command_names\":";
        appendJsonStringArray(out, commandNames);
        out << ",\"simple_deterministic_row_rule\":" << (simpleDeterministicRowRule ? "true" : "false")
            << ",\"delta_hint\":{\"dx\":" << dx << ",\"dy\":" << dy << "}"
            << ",\"rule_object_ids\":";
        appendObjectIdsFromMask(out, game, rule.ruleMask);
        out << ",\"rule_movement_bits\":";
        appendMovementBitPairs(out, game, rule.ruleMovementMask);
        out << ",\"rows\":[";
        for (size_t rowIndex = 0; rowIndex < rule.patterns.size(); ++rowIndex) {
            if (rowIndex != 0) {
                out << ",";
            }
            const auto& row = rule.patterns[rowIndex];
            const int32_t ellipsisCount = rowIndex < rule.ellipsisCount.size() ? rule.ellipsisCount[rowIndex] : 0;
            const puzzlescript::MaskOffset rowObjectOffset = rowIndex < rule.cellRowMasksCount
                ? game.cellRowMaskOffsets[static_cast<size_t>(rule.cellRowMasksFirst + rowIndex)]
                : puzzlescript::kNullMaskOffset;
            const puzzlescript::MaskOffset rowMovementOffset = rowIndex < rule.cellRowMasksMovementsCount
                ? game.cellRowMaskMovementsOffsets[static_cast<size_t>(rule.cellRowMasksMovementsFirst + rowIndex)]
                : puzzlescript::kNullMaskOffset;

            int32_t concreteCellCount = 0;
            int32_t finalEllipsisIndex = -1;
            for (size_t cellIndex = 0; cellIndex < row.size(); ++cellIndex) {
                const bool concrete = row[cellIndex].kind != puzzlescript::Pattern::Kind::Ellipsis;
                if (concrete) {
                    ++concreteCellCount;
                } else {
                    finalEllipsisIndex = static_cast<int32_t>(cellIndex);
                }
            }
            int32_t minConcreteSuffix = concreteCellCount;
            if (finalEllipsisIndex >= 0) {
                minConcreteSuffix = 0;
                for (size_t cellIndex = static_cast<size_t>(finalEllipsisIndex + 1); cellIndex < row.size(); ++cellIndex) {
                    if (row[cellIndex].kind != puzzlescript::Pattern::Kind::Ellipsis) {
                        ++minConcreteSuffix;
                    }
                }
            }

            std::vector<int32_t> concreteAnchorObjectIds;
            std::vector<std::vector<int32_t>> anyAnchorObjectIds;
            for (const auto& pattern : row) {
                if (pattern.kind == puzzlescript::Pattern::Kind::Ellipsis) {
                    continue;
                }
                concreteAnchorObjectIds.insert(
                    concreteAnchorObjectIds.end(),
                    pattern.objectAnchorIds.begin(),
                    pattern.objectAnchorIds.end()
                );
                anyAnchorObjectIds.insert(
                    anyAnchorObjectIds.end(),
                    pattern.anyObjectAnchorIds.begin(),
                    pattern.anyObjectAnchorIds.end()
                );
            }

            out << "{\"row_index\":" << rowIndex
                << ",\"ellipsis_count\":" << ellipsisCount
                << ",\"object_ids\":";
            appendObjectIdsFromMask(out, game, rowObjectOffset);
            out << ",\"movement_bits\":";
            appendMovementBitPairs(out, game, rowMovementOffset);
            out << ",\"concrete_anchor_object_ids\":";
            appendJsonIntArrayUnique(out, std::move(concreteAnchorObjectIds));
            out << ",\"any_anchor_object_ids\":[";
            for (size_t anyIndex = 0; anyIndex < anyAnchorObjectIds.size(); ++anyIndex) {
                if (anyIndex != 0) {
                    out << ",";
                }
                appendJsonIntArrayUnique(out, std::move(anyAnchorObjectIds[anyIndex]));
            }
            out << "],\"concrete_cell_count\":" << concreteCellCount
                << ",\"min_concrete_suffix\":" << minConcreteSuffix
                << ",\"scan_order\":\"x_major\"}";
        }
        out << "],\"replacements\":[";
        bool firstReplacement = true;
        for (size_t rowIndex = 0; rowIndex < rule.patterns.size(); ++rowIndex) {
            const auto& row = rule.patterns[rowIndex];
            for (size_t cellIndex = 0; cellIndex < row.size(); ++cellIndex) {
                const auto& pattern = row[cellIndex];
                if (!pattern.replacement.has_value()) {
                    continue;
                }
                if (replacementGuaranteedNoop(game, *pattern.replacement, pattern)) {
                    continue;
                }
                if (!firstReplacement) {
                    out << ",";
                }
                firstReplacement = false;
                const auto& repl = *pattern.replacement;
                const bool touchesObjects =
                    maskHasAnyBit(game, repl.objectsClear, game.wordCount)
                    || maskHasAnyBit(game, repl.objectsSet, game.wordCount)
                    || repl.hasRandomEntityMask;
                const bool touchesMovements =
                    maskHasAnyBit(game, repl.movementsClear, game.movementWordCount)
                    || maskHasAnyBit(game, repl.movementsSet, game.movementWordCount)
                    || repl.hasMovementsLayerMask
                    || repl.hasRandomDirMask;
                const bool touchesRandom = repl.hasRandomEntityMask || repl.hasRandomDirMask;
                const bool touchesRigid = rule.rigid && touchesMovements;
                out << "{\"row_index\":" << rowIndex
                    << ",\"cell_index\":" << cellIndex
                    << ",\"touches_objects\":" << (touchesObjects ? "true" : "false")
                    << ",\"touches_movements\":" << (touchesMovements ? "true" : "false")
                    << ",\"touches_movements_layer\":" << (repl.hasMovementsLayerMask ? "true" : "false")
                    << ",\"touches_random\":" << (touchesRandom ? "true" : "false")
                    << ",\"touches_random_entity\":" << (repl.hasRandomEntityMask ? "true" : "false")
                    << ",\"touches_random_dir\":" << (repl.hasRandomDirMask ? "true" : "false")
                    << ",\"touches_rigid\":" << (touchesRigid ? "true" : "false")
                    << ",\"simple_direct_mask\":" << ((!touchesRandom && !touchesRigid) ? "true" : "false")
                    << ",\"objects_clear_ids\":";
                appendObjectIdsFromMask(out, game, repl.objectsClear);
                out << ",\"objects_set_ids\":";
                appendObjectIdsFromMask(out, game, repl.objectsSet);
                out << ",\"movements_clear_bits\":";
                appendMovementBitPairs(out, game, repl.movementsClear);
                out << ",\"movements_set_bits\":";
                appendMovementBitPairs(out, game, repl.movementsSet);
                out << ",\"movements_layer_bits\":";
                appendMovementBitPairs(out, game, repl.movementsLayerMask);
                out << ",\"random_dir_bits\":";
                appendMovementBitPairs(out, game, repl.randomDirMask);
                out << ",\"random_entity_object_ids\":";
                appendObjectIdsFromMask(out, game, repl.randomEntityMask);
                out << ",\"random_entity_choices\":";
                appendJsonIntArray(out, repl.randomEntityChoices);
                out << ",\"random_dir_layers\":";
                appendJsonIntArray(out, repl.randomDirLayers);
                out << ",\"movement_layers\":";
                appendMovementLayersFromMask(out, game, repl.movementsLayerMask);
                out << "}";
            }
        }
        out << "]}";
    };

    auto appendGroups = [&](const std::vector<std::vector<puzzlescript::Rule>>& groups, bool late) {
        out << "[";
        size_t emittedGroupIndex = 0;
        for (size_t groupIndex = 0; groupIndex < groups.size(); ++groupIndex) {
            bool hasEmittedRules = false;
            for (const auto& rule : groups[groupIndex]) {
                if (!ruleImpossible(game, rule)) {
                    hasEmittedRules = true;
                    break;
                }
            }
            if (!hasEmittedRules) {
                continue;
            }
            if (emittedGroupIndex != 0) {
                out << ",";
            }
            out << "[";
            size_t emittedRuleIndex = 0;
            for (size_t ruleIndex = 0; ruleIndex < groups[groupIndex].size(); ++ruleIndex) {
                const auto& rule = groups[groupIndex][ruleIndex];
                if (ruleImpossible(game, rule)) {
                    continue;
                }
                if (emittedRuleIndex != 0) {
                    out << ",";
                }
                appendRuleEntry(rule, emittedGroupIndex, emittedRuleIndex, late);
                ++emittedRuleIndex;
            }
            out << "]";
            ++emittedGroupIndex;
        }
        out << "]";
    };
    out << "{\"schema_version\":1,\"rules\":";
    appendGroups(game.rules, false);
    out << ",\"late_rules\":";
    appendGroups(game.lateRules, true);
    out << "}";
}

std::string serializeRuntimeGameDebugJson(const puzzlescript::LoadedGame& loadedGame) {
    const puzzlescript::Game& game = *loadedGame.information;
    const puzzlescript::MetaGameState& initialMetaGameState = loadedGame.initialMetaGameState;
    std::ostringstream out;
    // Emit JSON that is loadable by puzzlescript::loadGameFromJson.
    out << "{\n";
    out << "  \"schema_version\": " << game.schemaVersion << ",\n";
    out << "  \"document\": {\"command\":[\"loadLevel\",0],\"error_count\":0,\"errors\":[],\"input_file\":\"\",\"random_seed\":\"\"},\n";
    out << "  \"game\": {\n";
    out << "    \"strides\": {\"object\": " << game.strideObject << ", \"movement\": " << game.strideMovement
        << ", \"layers\": " << game.layerCount << "},\n";
    out << "    \"object_count\": " << game.objectCount << ",\n";
    out << "    \"colors\": {\"foreground\": " << jsonStringLiteral(game.foregroundColor)
        << ", \"background\": " << jsonStringLiteral(game.backgroundColor) << "},\n";
    out << "    \"background\": {\"id\": " << game.backgroundId << ", \"layer\": " << game.backgroundLayer << "},\n";
    out << "    \"metadata_pairs\": "; appendJsonStringArray(out, game.metadata.pairs); out << ",\n";
    // metadata_map / lines are optional; include to ease diffs.
    out << "    \"metadata_map\": {";
    {
        bool first = true;
        for (const auto& [k, v] : game.metadata.values) {
            if (!first) out << ",";
            first = false;
            out << jsonStringLiteral(k) << ":" << jsonStringLiteral(v);
        }
    }
    out << "},\n";
    out << "    \"metadata_lines\": {";
    {
        bool first = true;
        for (const auto& [k, v] : game.metadata.lines) {
            if (!first) out << ",";
            first = false;
            out << jsonStringLiteral(k) << ":" << v;
        }
    }
    out << "},\n";
    out << "    \"id_dict\": "; appendJsonStringArray(out, game.idDict); out << ",\n";
    out << "    \"collision_layers\": [";
    for (size_t layer = 0; layer < game.collisionLayers.size(); ++layer) {
        if (layer != 0) out << ",";
        appendJsonStringArray(out, game.collisionLayers[layer]);
    }
    out << "],\n";
    out << "    \"layer_masks\": [";
    for (size_t layer = 0; layer < game.layerMaskOffsets.size(); ++layer) {
        if (layer != 0) out << ",";
        appendJsonMask(out, game, game.layerMaskOffsets[layer], game.wordCount);
    }
    out << "],\n";
    out << "    \"player_mask\": {\"aggregate\": " << (game.playerMaskAggregate ? "true" : "false") << ", \"mask\":";
    appendJsonMask(out, game, game.playerMask, game.wordCount);
    out << "},\n";

    out << "    \"objects\": [";
    for (size_t idx = 0; idx < game.objectsById.size(); ++idx) {
        if (idx != 0) out << ",";
        const auto& obj = game.objectsById[idx];
        out << "{"
            << "\"name\":" << jsonStringLiteral(obj.name)
            << ",\"id\":" << obj.id
            << ",\"layer\":" << obj.layer
            << ",\"colors\":"; appendJsonStringArray(out, obj.colors);
        out << ",\"spritematrix\":[";
        for (size_t r = 0; r < obj.sprite.size(); ++r) {
            if (r != 0) out << ",";
            appendJsonIntArray(out, obj.sprite[r]);
        }
        out << "]"
            << "}";
    }
    out << "],\n";

    out << "    \"levels\": [";
    for (size_t levelIndex = 0; levelIndex < game.levels.size(); ++levelIndex) {
        if (levelIndex != 0) out << ",";
        const auto& level = game.levels[levelIndex];
        if (level.isMessage) {
            out << "{\"kind\":\"message\",\"message\":" << jsonStringLiteral(level.message) << "}";
        } else {
            out << "{\"kind\":\"level\",\"line_number\":" << level.lineNumber << ",\"width\":" << level.width
                << ",\"height\":" << level.height << ",\"layer_count\":" << level.layerCount << ",\"objects\":";
            appendJsonIntArray(out, level.objects);
            out << "}";
        }
    }
    out << "],\n";
    auto appendLoopPointTable = [&](const puzzlescript::LoopPointTable& table) {
        out << "{";
        bool first = true;
        for (size_t index = 0; index < table.entries.size(); ++index) {
            if (!table.entries[index].has_value()) {
                continue;
            }
            if (!first) {
                out << ",";
            }
            first = false;
            out << jsonStringLiteral(std::to_string(index)) << ":" << *table.entries[index];
        }
        out << "}";
    };
    auto appendRules = [&](const std::vector<std::vector<puzzlescript::Rule>>& groups) {
        out << "[";
        for (size_t groupIndex = 0; groupIndex < groups.size(); ++groupIndex) {
            if (groupIndex != 0) out << ",";
            out << "[";
            for (size_t ruleIndex = 0; ruleIndex < groups[groupIndex].size(); ++ruleIndex) {
                if (ruleIndex != 0) out << ",";
                const auto& rule = groups[groupIndex][ruleIndex];
                out << "{\"direction\":" << rule.direction << ",\"has_replacements\":" << (rule.hasReplacements ? "true" : "false")
                    << ",\"line_number\":" << rule.lineNumber << ",\"group_number\":" << rule.groupNumber
                    << ",\"rigid\":" << (rule.rigid ? "true" : "false") << ",\"is_random\":" << (rule.isRandom ? "true" : "false")
                    << ",\"ellipsis_count\":";
                appendJsonIntArray(out, rule.ellipsisCount);
                out << ",\"commands\":[";
                for (size_t commandIndex = 0; commandIndex < rule.commands.size(); ++commandIndex) {
                    if (commandIndex != 0) {
                        out << ",";
                    }
                    const auto& command = rule.commands[commandIndex];
                    out << "[" << jsonStringLiteral(command.name);
                    if (command.argument.has_value()) {
                        out << "," << jsonStringLiteral(*command.argument);
                    }
                    out << "]";
                }
                out << "]";
                out << ",\"cell_row_masks\":[";
                for (uint32_t row = 0; row < rule.cellRowMasksCount; ++row) {
                    if (row != 0) out << ",";
                    const auto offset = game.cellRowMaskOffsets[static_cast<size_t>(rule.cellRowMasksFirst + row)];
                    appendJsonMask(out, game, offset, game.wordCount);
                }
                out << "],\"cell_row_masks_movements\":[";
                for (uint32_t row = 0; row < rule.cellRowMasksMovementsCount; ++row) {
                    if (row != 0) out << ",";
                    const auto offset = game.cellRowMaskMovementsOffsets[static_cast<size_t>(rule.cellRowMasksMovementsFirst + row)];
                    appendJsonMask(out, game, offset, game.movementWordCount);
                }
                out << "],\"rule_mask\":"; appendJsonMask(out, game, rule.ruleMask, game.wordCount);
                out << ",\"patterns\":[";
                for (size_t rowIndex = 0; rowIndex < rule.patterns.size(); ++rowIndex) {
                    if (rowIndex != 0) out << ",";
                    out << "[";
                    for (size_t patternIndex = 0; patternIndex < rule.patterns[rowIndex].size(); ++patternIndex) {
                        if (patternIndex != 0) out << ",";
                        const auto& pattern = rule.patterns[rowIndex][patternIndex];
                        if (pattern.kind == puzzlescript::Pattern::Kind::Ellipsis) {
                            out << "{\"kind\":\"ellipsis\"}";
                        } else {
                            out << "{\"kind\":\"cell_pattern\",\"objects_present\":";
                            appendJsonMask(out, game, pattern.objectsPresent, game.wordCount);
                            out << ",\"objects_missing\":"; appendJsonMask(out, game, pattern.objectsMissing, game.wordCount);
                            out << ",\"movements_present\":"; appendJsonMask(out, game, pattern.movementsPresent, game.movementWordCount);
                            out << ",\"movements_missing\":"; appendJsonMask(out, game, pattern.movementsMissing, game.movementWordCount);
                            out << ",\"any_objects_present\":[";
                            for (uint32_t anyIndex = 0; anyIndex < pattern.anyObjectsCount; ++anyIndex) {
                                if (anyIndex != 0) out << ",";
                                const auto offset = game.anyObjectOffsets[static_cast<size_t>(pattern.anyObjectsFirst + anyIndex)];
                                appendJsonMask(out, game, offset, game.wordCount);
                            }
                            out << "]";
                            if (pattern.replacement.has_value()) {
                                const auto& repl = *pattern.replacement;
                                out << ",\"replacement\":{\"objects_clear\":"; appendJsonMask(out, game, repl.objectsClear, game.wordCount);
                                out << ",\"objects_set\":"; appendJsonMask(out, game, repl.objectsSet, game.wordCount);
                                out << ",\"movements_clear\":"; appendJsonMask(out, game, repl.movementsClear, game.movementWordCount);
                                out << ",\"movements_set\":"; appendJsonMask(out, game, repl.movementsSet, game.movementWordCount);
                                out << ",\"movements_layer_mask\":"; appendJsonMask(out, game, repl.movementsLayerMask, game.movementWordCount);
                                out << ",\"random_entity_mask\":"; appendJsonMask(out, game, repl.randomEntityMask, game.wordCount);
                                out << ",\"random_dir_mask\":"; appendJsonMask(out, game, repl.randomDirMask, game.movementWordCount);
                                out << "}";
                            } else {
                                out << ",\"replacement\":null";
                            }
                            out << "}";
                        }
                    }
                    out << "]";
                }
                out << "]}";
            }
            out << "]";
        }
        out << "]";
    };
    out << "    \"rules\": "; appendRules(game.rules); out << ",\n";
    out << "    \"late_rules\": "; appendRules(game.lateRules); out << ",\n";
    out << "    \"rule_plan_v1\": "; appendRulePlanJson(out, game); out << ",\n";
    out << "    \"sfx_events\": {";
    {
        bool first = true;
        for (const auto& [name, seed] : game.sfxEvents) {
            if (!first) out << ",";
            first = false;
            out << jsonStringLiteral(name) << ":" << seed;
        }
    }
    out << "},\n";
    auto appendSoundMaskEntries = [&](const std::vector<puzzlescript::SoundMaskEntry>& entries) {
        out << "[";
        for (size_t index = 0; index < entries.size(); ++index) {
            if (index != 0) out << ",";
            const auto& entry = entries[index];
            out << "{\"objectMask\":";
            appendJsonMask(out, game, entry.objectMask, game.wordCount);
            out << ",\"directionMask\":";
            appendJsonMask(out, game, entry.directionMask, entry.directionMaskWidth);
            out << ",\"seed\":" << entry.seed << "}";
        }
        out << "]";
    };
    out << "    \"sfx_creation_masks\": "; appendSoundMaskEntries(game.sfxCreationMasks); out << ",\n";
    out << "    \"sfx_destruction_masks\": "; appendSoundMaskEntries(game.sfxDestructionMasks); out << ",\n";
    out << "    \"sfx_movement_masks\": [";
    for (size_t layer = 0; layer < game.sfxMovementMasks.size(); ++layer) {
        if (layer != 0) out << ",";
        appendSoundMaskEntries(game.sfxMovementMasks[layer]);
    }
    out << "],\n";
    out << "    \"sfx_movement_failure_masks\": "; appendSoundMaskEntries(game.sfxMovementFailureMasks); out << ",\n";
    out << "    \"loop_point\": "; appendLoopPointTable(game.loopPoint); out << ",\n";
    out << "    \"late_loop_point\": "; appendLoopPointTable(game.lateLoopPoint); out << "\n";
    out << "  }";

    // prepared_session is optional, but emitting it makes native-vs-js diffs much easier.
    out << ",\n  \"prepared_session\": {";
    out << "\"current_level_index\":" << initialMetaGameState.currentLevelIndex << ",";
    out << "\"current_level_target\":null,";
    out << "\"title_screen\":" << (initialMetaGameState.titleScreen ? "true" : "false") << ",";
    out << "\"text_mode\":" << (initialMetaGameState.textMode ? "true" : "false") << ",";
    out << "\"title_mode\":" << initialMetaGameState.titleMode << ",";
    out << "\"title_selection\":" << initialMetaGameState.titleSelection << ",";
    out << "\"title_selected\":" << (initialMetaGameState.titleSelected ? "true" : "false") << ",";
    out << "\"message_selected\":" << (initialMetaGameState.messageSelected ? "true" : "false") << ",";
    out << "\"winning\":" << (initialMetaGameState.winning ? "true" : "false") << ",";
    out << "\"loaded_level_seed\":" << jsonStringLiteral(initialMetaGameState.loadedLevelSeed) << ",";
    out << "\"random_state\":null,";
    out << "\"old_flickscreen_dat\":[],";
    out << "\"level\":";
    if (initialMetaGameState.level.isMessage) {
        out << "{\"kind\":\"message\",\"message\":" << jsonStringLiteral(initialMetaGameState.level.message) << "}";
    } else {
        out << "{\"kind\":\"level\",\"line_number\":" << initialMetaGameState.level.lineNumber
            << ",\"width\":" << initialMetaGameState.level.width
            << ",\"height\":" << initialMetaGameState.level.height
            << ",\"layer_count\":" << initialMetaGameState.level.layerCount
            << ",\"objects\":";
        appendJsonIntArray(out, initialMetaGameState.level.objects);
        out << "}";
    }
    out << ",\"serialized_level\":" << jsonStringLiteral(initialMetaGameState.serializedLevel);
    out << ",\"restart_target\":null";
    out << "}\n";
    out << "}\n";
    return out.str();
}

std::string cppStringLiteral(std::string_view value) {
    std::ostringstream out;
    out << '"';
    for (unsigned char ch : value) {
        switch (ch) {
            case '\\': out << "\\\\"; break;
            case '"': out << "\\\""; break;
            case '\n': out << "\\n"; break;
            case '\r': out << "\\r"; break;
            case '\t': out << "\\t"; break;
            default:
                if (ch < 0x20) {
                    static constexpr char kHex[] = "0123456789abcdef";
                    out << "\\x" << kHex[ch >> 4] << kHex[ch & 0x0f];
                } else {
                    out << static_cast<char>(ch);
                }
                break;
        }
    }
    out << '"';
    return out.str();
}

std::string safeCppIdentifier(std::string_view value) {
    std::string out;
    out.reserve(value.size() + 1);
    for (const unsigned char ch : value) {
        if ((ch >= 'a' && ch <= 'z') || (ch >= 'A' && ch <= 'Z') || (ch >= '0' && ch <= '9')) {
            out.push_back(static_cast<char>(ch));
        } else {
            out.push_back('_');
        }
    }
    if (out.empty() || (out[0] >= '0' && out[0] <= '9')) {
        out.insert(out.begin(), '_');
    }
    return out;
}

bool isCompilableReplacement(const puzzlescript::Replacement& replacement) {
    return !replacement.hasRandomEntityMask && !replacement.hasRandomDirMask;
}

struct CompiledRulesOptions {
    size_t maxRows = 1;
};

std::string compiledRuleMissReason(
    const puzzlescript::Rule& rule,
    const CompiledRulesOptions& options,
    bool allowRandomRule = false
) {
    if (rule.isRandom && !allowRandomRule) {
        return "random_rule";
    }
    if (rule.patterns.empty()) {
        return "empty_row";
    }
    if (rule.patterns.size() > options.maxRows) {
        return "row_limit";
    }
    if (rule.ellipsisCount.size() < rule.patterns.size()) {
        return "missing_ellipsis_metadata";
    }
    for (size_t rowIndex = 0; rowIndex < rule.patterns.size(); ++rowIndex) {
        if (rule.patterns[rowIndex].empty()) {
            return "empty_row";
        }
        for (const auto& pattern : rule.patterns[rowIndex]) {
            if (pattern.kind == puzzlescript::Pattern::Kind::Ellipsis) {
                continue;
            }
            if (pattern.kind != puzzlescript::Pattern::Kind::CellPattern) {
                return "non_cell_pattern";
            }
        }
    }
    return {};
}

bool isCompilableRule(const puzzlescript::Rule& rule, const CompiledRulesOptions& options) {
    return compiledRuleMissReason(rule, options).empty();
}

std::string compiledGroupMissReason(const std::vector<puzzlescript::Rule>& group, const CompiledRulesOptions& options) {
    if (group.empty()) {
        return "empty_group";
    }
    const bool randomGroup = group[0].isRandom;
    for (const auto& rule : group) {
        const std::string reason = compiledRuleMissReason(rule, options, randomGroup);
        if (!reason.empty()) {
            return reason;
        }
    }
    return {};
}

bool isCompilableGroup(const std::vector<puzzlescript::Rule>& group, const CompiledRulesOptions& options) {
    return compiledGroupMissReason(group, options).empty();
}

bool areAllGroupsCompilable(const std::vector<std::vector<puzzlescript::Rule>>& groups, const CompiledRulesOptions& options) {
    return std::all_of(groups.begin(), groups.end(), [&](const std::vector<puzzlescript::Rule>& group) {
        return isCompilableGroup(group, options);
    });
}

bool isKnownSpecializedFullTurnCommandName(std::string_view name) {
    if (name == "again"
        || name == "cancel"
        || name == "checkpoint"
        || name == "message"
        || name == "restart"
        || name == "win") {
        return true;
    }
    if (name.size() <= 3 || name.substr(0, 3) != "sfx") {
        return false;
    }
    return std::all_of(name.begin() + 3, name.end(), [](const char ch) {
        return ch >= '0' && ch <= '9';
    });
}

std::string specializedFullTurnCommandStatusForGroups(const std::vector<std::vector<puzzlescript::Rule>>& groups) {
    bool sawCommand = false;
    for (const auto& group : groups) {
        for (const auto& rule : group) {
            for (const auto& command : rule.commands) {
                sawCommand = true;
                if (!isKnownSpecializedFullTurnCommandName(command.name)) {
                    return "unknown_interpreter";
                }
            }
        }
    }
    return sawCommand ? "generated_queue_interpreter_tail" : "none";
}

std::string specializedFullTurnCommandStatus(const puzzlescript::Game& game) {
    const std::string earlyStatus = specializedFullTurnCommandStatusForGroups(game.rules);
    const std::string lateStatus = specializedFullTurnCommandStatusForGroups(game.lateRules);
    if (earlyStatus == "unknown_interpreter" || lateStatus == "unknown_interpreter") {
        return "unknown_interpreter";
    }
    if (earlyStatus == "generated_queue_interpreter_tail" || lateStatus == "generated_queue_interpreter_tail") {
        return "generated_queue_interpreter_tail";
    }
    return "none";
}

std::optional<std::string_view> compiledRuleCommandKindExpression(std::string_view commandName) {
    if (commandName == "again") return "CompiledRuleCommandKind::Again";
    if (commandName == "cancel") return "CompiledRuleCommandKind::Cancel";
    if (commandName == "checkpoint") return "CompiledRuleCommandKind::Checkpoint";
    if (commandName == "message") return "CompiledRuleCommandKind::Message";
    if (commandName == "restart") return "CompiledRuleCommandKind::Restart";
    if (commandName == "win") return "CompiledRuleCommandKind::Win";
    if (isKnownSpecializedFullTurnCommandName(commandName)
        && commandName.size() > 3
        && commandName.substr(0, 3) == "sfx") {
        return "CompiledRuleCommandKind::Output";
    }
    return std::nullopt;
}

bool canGenerateCompiledRuleCommandQueue(const puzzlescript::Rule& rule) {
    return std::all_of(rule.commands.begin(), rule.commands.end(), [](const puzzlescript::RuleCommand& command) {
        return compiledRuleCommandKindExpression(command.name).has_value();
    });
}

struct SpecializedFullTurnSupport {
    // These are the generated top-level early/late rule loops: they sequence
    // rulegroups and preserve BEGIN LOOP / END LOOP loop-point jumps.
    bool earlyRuleLoopsGenerated = false;
    bool lateRuleLoopsGenerated = false;
    std::string commandStatus = "unknown_interpreter";
    bool wholeTurnSupported = false;
    std::string wholeTurnFallbackReason = "interpreter_delegation";
};

struct CompactTurnSupport {
    bool supported = false;
    std::string fallbackReason = "interpreter_delegation";
    bool interpreterBridge = false;
    std::string nativeFallbackReason = "interpreter_delegation";
};

CompactTurnSupport compactNativeTurnSupportForGame(const puzzlescript::Game& game) {
    (void)game;
    CompactTurnSupport support;
    support.fallbackReason = "native_compact_generator_rebuild";
    support.nativeFallbackReason = support.fallbackReason;
    return support;
}

CompactTurnSupport compactTurnSupportForGame(const puzzlescript::Game& game) {
    CompactTurnSupport support = compactNativeTurnSupportForGame(game);
    support.nativeFallbackReason = support.fallbackReason;
    if (!support.supported) {
        support.supported = true;
        support.interpreterBridge = true;
        support.fallbackReason = "interpreter_bridge";
    }
    return support;
}

SpecializedFullTurnSupport specializedFullTurnSupportForGame(
    const puzzlescript::Game& game,
    const CompiledRulesOptions& options
) {
    SpecializedFullTurnSupport support;
    support.earlyRuleLoopsGenerated = areAllGroupsCompilable(game.rules, options);
    support.lateRuleLoopsGenerated = areAllGroupsCompilable(game.lateRules, options);
    support.commandStatus = specializedFullTurnCommandStatus(game);

    if (!support.earlyRuleLoopsGenerated) {
        support.wholeTurnFallbackReason = "early_rule_loops_interpreter";
    } else if (!support.lateRuleLoopsGenerated) {
        support.wholeTurnFallbackReason = "late_rule_loops_interpreter";
    } else if (support.commandStatus == "unknown_interpreter") {
        support.wholeTurnFallbackReason = "unsupported_command";
    } else {
        support.wholeTurnFallbackReason = "movement_interpreter";
    }
    return support;
}

SpecializedFullTurnSupport specializedFullTurnSupportForMissingGame() {
    return SpecializedFullTurnSupport{};
}

bool ruleHasEllipsis(const puzzlescript::Rule& rule) {
    for (const int32_t count : rule.ellipsisCount) {
        if (count != 0) {
            return true;
        }
    }
    for (const auto& row : rule.patterns) {
        for (const auto& pattern : row) {
            if (pattern.kind == puzzlescript::Pattern::Kind::Ellipsis) {
                return true;
            }
        }
    }
    return false;
}

bool ruleHasRandomReplacement(const puzzlescript::Rule& rule) {
    for (const auto& row : rule.patterns) {
        for (const auto& pattern : row) {
            if (pattern.replacement.has_value() && !isCompilableReplacement(*pattern.replacement)) {
                return true;
            }
        }
    }
    return false;
}

bool ruleUsesRuntimeRowHelpers(const puzzlescript::Rule& rule) {
    return rule.rigid || ruleHasEllipsis(rule) || ruleHasRandomReplacement(rule);
}

std::string compiledMaskWordLiteral(puzzlescript::MaskWord word) {
    std::ostringstream out;
    out << "static_cast<MaskWord>(static_cast<MaskWordUnsigned>("
        << static_cast<puzzlescript::MaskWordUnsigned>(word)
        << "ULL))";
    return out.str();
}

std::vector<puzzlescript::MaskWord> compiledMaskWords(
    const puzzlescript::Game& game,
    puzzlescript::MaskOffset offset,
    uint32_t wordCount
) {
    std::vector<puzzlescript::MaskWord> words(static_cast<size_t>(wordCount), 0);
    if (offset == puzzlescript::kNullMaskOffset || wordCount == 0) {
        return words;
    }
    const size_t begin = static_cast<size_t>(offset);
    for (uint32_t word = 0; word < wordCount; ++word) {
        const size_t index = begin + static_cast<size_t>(word);
        if (index < game.maskArena.size()) {
            words[static_cast<size_t>(word)] = game.maskArena[index];
        }
    }
    return words;
}

void emitInlineMaskBitsSetCheck(
    std::ostream& out,
    const puzzlescript::Game& game,
    puzzlescript::MaskOffset offset,
    uint32_t wordCount,
    const std::string& actualExpr,
    const std::string& indent = "    ",
    const std::string& failStatement = "return false"
) {
    if (offset == puzzlescript::kNullMaskOffset || wordCount == 0) {
        return;
    }
    const auto words = compiledMaskWords(game, offset, wordCount);
    bool anyRequired = false;
    for (uint32_t word = 0; word < wordCount; ++word) {
        const auto mask = words[static_cast<size_t>(word)];
        if (mask == 0) {
            continue;
        }
        anyRequired = true;
        out << indent << "if (((" << actualExpr << ")[" << word << "] & "
            << compiledMaskWordLiteral(mask) << ") != "
            << compiledMaskWordLiteral(mask) << ") " << failStatement << ";\n";
    }
    if (!anyRequired) {
        out << indent << "(void)" << actualExpr << ";\n";
    }
}

void emitInlineMaskAnyBitsReject(
    std::ostream& out,
    const puzzlescript::Game& game,
    puzzlescript::MaskOffset offset,
    uint32_t wordCount,
    const std::string& actualExpr,
    const std::string& indent = "    ",
    const std::string& failStatement = "return false"
) {
    if (offset == puzzlescript::kNullMaskOffset || wordCount == 0) {
        return;
    }
    const auto words = compiledMaskWords(game, offset, wordCount);
    for (uint32_t word = 0; word < wordCount; ++word) {
        const auto mask = words[static_cast<size_t>(word)];
        if (mask == 0) {
            continue;
        }
        out << indent << "if ((((" << actualExpr << ")[" << word << "] & "
            << compiledMaskWordLiteral(mask) << ") != 0)) " << failStatement << ";\n";
    }
}

void emitMaskBitsSetCheck(
    std::ostream& out,
    const puzzlescript::Game& game,
    const std::string& maskName,
    puzzlescript::MaskOffset offset,
    uint32_t wordCount,
    const std::string& actualExpr,
    const std::string& actualCountExpr
) {
    if (offset == puzzlescript::kNullMaskOffset || wordCount == 0) {
        return;
    }
    emitInlineMaskBitsSetCheck(out, game, offset, wordCount, actualExpr);
    (void)maskName;
    (void)actualCountExpr;
}

void emitMaskAnyBitsReject(
    std::ostream& out,
    const puzzlescript::Game& game,
    const std::string& maskName,
    puzzlescript::MaskOffset offset,
    uint32_t wordCount,
    const std::string& actualExpr,
    const std::string& actualCountExpr
) {
    if (offset == puzzlescript::kNullMaskOffset || wordCount == 0) {
        return;
    }
    emitInlineMaskAnyBitsReject(out, game, offset, wordCount, actualExpr);
    (void)maskName;
    (void)actualCountExpr;
}

void emitPatternPredicate(
    std::ostream& out,
    const puzzlescript::Game& game,
    const puzzlescript::Pattern& pattern,
    const std::string& suffix,
    const std::string& tileExpr
) {
    out << "    const int32_t tileIdx" << suffix << " = static_cast<int32_t>(" << tileExpr << ");\n"
        << "    const MaskWord* objects" << suffix << " = compiledRuleCellObjects(session, tileIdx" << suffix << ");\n"
        << "    const MaskWord* movements" << suffix << " = compiledRuleCellMovements(session, tileIdx" << suffix << ");\n";
    if (pattern.hasObjectsPresent) {
        emitMaskBitsSetCheck(out, game, "objectsPresent" + suffix, pattern.objectsPresent, game.wordCount, "objects" + suffix, std::to_string(game.wordCount) + "U");
    }
    if (pattern.hasObjectsMissing) {
        emitMaskAnyBitsReject(out, game, "objectsMissing" + suffix, pattern.objectsMissing, game.wordCount, "objects" + suffix, std::to_string(game.wordCount) + "U");
    }
    for (uint32_t index = 0; index < pattern.anyObjectsCount; ++index) {
        const auto offsetIndex = static_cast<size_t>(pattern.anyObjectsFirst + index);
        if (offsetIndex >= game.anyObjectOffsets.size()) {
            continue;
        }
        const auto anyWords = compiledMaskWords(game, game.anyObjectOffsets[offsetIndex], game.wordCount);
        out << "    if (!(";
        bool emitted = false;
        for (uint32_t word = 0; word < game.wordCount; ++word) {
            const auto mask = anyWords[static_cast<size_t>(word)];
            if (mask == 0) {
                continue;
            }
            if (emitted) {
                out << " || ";
            }
            out << "((objects" << suffix << "[" << word << "] & " << compiledMaskWordLiteral(mask) << ") != 0)";
            emitted = true;
        }
        out << (emitted ? ")) return false;\n" : "false)) return false;\n");
    }
    if (pattern.hasMovementsPresent) {
        emitMaskBitsSetCheck(out, game, "movementsPresent" + suffix, pattern.movementsPresent, game.movementWordCount, "movements" + suffix, std::to_string(game.movementWordCount) + "U");
    }
    if (pattern.hasMovementsMissing) {
        emitMaskAnyBitsReject(out, game, "movementsMissing" + suffix, pattern.movementsMissing, game.movementWordCount, "movements" + suffix, std::to_string(game.movementWordCount) + "U");
    }
}

void emitReplacementApply(
    std::ostream& out,
    const puzzlescript::Game& game,
    const puzzlescript::Replacement& replacement,
    const std::string& suffix,
    const std::string& tileExpr
) {
    const auto objectsClear = compiledMaskWords(game, replacement.objectsClear, game.wordCount);
    const auto objectsSet = compiledMaskWords(game, replacement.objectsSet, game.wordCount);
    const auto movementsClear = compiledMaskWords(game, replacement.movementsClear, game.movementWordCount);
    const auto movementsSet = compiledMaskWords(game, replacement.movementsSet, game.movementWordCount);
    const auto movementsLayerMask = replacement.hasMovementsLayerMask
        ? compiledMaskWords(game, replacement.movementsLayerMask, game.movementWordCount)
        : std::vector<puzzlescript::MaskWord>(static_cast<size_t>(game.movementWordCount), 0);
    const auto anyNonZero = [](const std::vector<puzzlescript::MaskWord>& words) {
        return std::any_of(words.begin(), words.end(), [](puzzlescript::MaskWord word) {
            return word != 0;
        });
    };
    const bool hasObjectEffect = anyNonZero(objectsClear) || anyNonZero(objectsSet);
    const bool hasMovementEffect = anyNonZero(movementsClear) || anyNonZero(movementsSet) || anyNonZero(movementsLayerMask);
    out << "    {\n"
        << "        const int32_t tile = " << tileExpr << ";\n"
        << "        bool objectsChanged = false;\n"
        << "        bool movementsChanged = false;\n";
    if (hasObjectEffect) {
        out << "        const MaskWord* oldObjects = compiledRuleCellObjects(session, tile);\n"
            << "        MaskWord newObjects[" << game.wordCount << "];\n"
            << "        MaskWord created[" << game.wordCount << "];\n"
            << "        MaskWord destroyed[" << game.wordCount << "];\n";
        for (uint32_t word = 0; word < game.wordCount; ++word) {
            out << "        {\n"
                << "            const MaskWord before = oldObjects[" << word << "];\n"
                << "            const MaskWord after = (before & ~" << compiledMaskWordLiteral(objectsClear[static_cast<size_t>(word)])
                << ") | " << compiledMaskWordLiteral(objectsSet[static_cast<size_t>(word)]) << ";\n"
                << "            newObjects[" << word << "] = after;\n"
                << "            created[" << word << "] = after & ~before;\n"
                << "            destroyed[" << word << "] = before & ~after;\n"
                << "            objectsChanged = objectsChanged || after != before;\n"
                << "        }\n";
        }
        if (game.wordCount == 1) {
            out << "        if (objectsChanged) compiledRuleSetCellObjectsWord1(session, tile, newObjects[0], created[0], destroyed[0]);\n";
        } else {
            out << "        if (objectsChanged) compiledRuleSetCellObjectsFromWords(session, tile, newObjects, created, destroyed);\n";
        }
    }
    if (hasMovementEffect) {
        out << "        const MaskWord* oldMovements = compiledRuleCellMovements(session, tile);\n"
            << "        MaskWord newMovements[" << game.movementWordCount << "];\n";
        for (uint32_t word = 0; word < game.movementWordCount; ++word) {
            const auto clearWord = static_cast<puzzlescript::MaskWord>(
                static_cast<puzzlescript::MaskWordUnsigned>(movementsClear[static_cast<size_t>(word)])
                | static_cast<puzzlescript::MaskWordUnsigned>(movementsLayerMask[static_cast<size_t>(word)])
            );
            out << "        {\n"
                << "            const MaskWord before = oldMovements[" << word << "];\n"
                << "            const MaskWord after = (before & ~" << compiledMaskWordLiteral(clearWord)
                << ") | " << compiledMaskWordLiteral(movementsSet[static_cast<size_t>(word)]) << ";\n"
                << "            newMovements[" << word << "] = after;\n"
                << "            movementsChanged = movementsChanged || after != before;\n"
                << "        }\n";
        }
        if (game.movementWordCount == 1) {
            out << "        if (movementsChanged) compiledRuleSetCellMovementsWord1(session, tile, newMovements[0]);\n";
        } else {
            out << "        if (movementsChanged) compiledRuleSetCellMovementsFromWords(session, tile, newMovements);\n";
        }
    }
    out << "        changed = changed || objectsChanged || movementsChanged;\n"
        << "    }\n";
    (void)suffix;
}

void emitRuleRowFunctions(
    std::ostream& out,
    const puzzlescript::Game& game,
    const puzzlescript::Rule& rule,
    const std::string& prefix,
    size_t rowIndex
) {
    const auto& row = rule.patterns[rowIndex];
    int32_t dx = 0;
    int32_t dy = 0;
    switch (rule.direction) {
        case 1: dy = -1; break;
        case 2: dy = 1; break;
        case 4: dx = -1; break;
        case 8: dx = 1; break;
        default: break;
    }

    out << "bool match_" << prefix << "_row" << rowIndex << "(FullState& session, int32_t startIndex) {\n"
        << "    const Game& game = *session.game;\n"
        << "    const int32_t delta = (" << dx << ") * currentLevelHeight(session) + (" << dy << ");\n";
    for (size_t cellIndex = 0; cellIndex < row.size(); ++cellIndex) {
        emitPatternPredicate(
            out,
            game,
            row[cellIndex],
            "_" + std::to_string(cellIndex),
            "startIndex + " + std::to_string(cellIndex) + " * delta"
        );
    }
    out << "    return true;\n"
        << "}\n\n";

    out << "bool apply_replacements_" << prefix << "_row" << rowIndex << "(FullState& session, int32_t startIndex) {\n"
        << "    const Game& game = *session.game;\n"
        << "    const int32_t delta = (" << dx << ") * currentLevelHeight(session) + (" << dy << ");\n"
        << "    bool changed = false;\n";
    for (size_t cellIndex = 0; cellIndex < row.size(); ++cellIndex) {
        if (!row[cellIndex].replacement.has_value()) {
            continue;
        }
        emitReplacementApply(
            out,
            game,
            *row[cellIndex].replacement,
            "_" + std::to_string(cellIndex),
            "startIndex + " + std::to_string(cellIndex) + " * delta"
        );
    }
    out << "    return changed;\n"
        << "}\n\n";
}

void emitCollectRowMatches(
    std::ostream& out,
    const puzzlescript::Game& game,
    const puzzlescript::Rule& rule,
    const std::string& prefix,
    size_t rowIndex,
    const std::string& matchesName,
    bool useSessionScratch
) {
    const auto& row = rule.patterns[rowIndex];
    const int32_t len = static_cast<int32_t>(row.size());
    const bool horizontal = rule.direction > 2;
    int32_t dx = 0;
    int32_t dy = 0;
    switch (rule.direction) {
        case 1: dy = -1; break;
        case 2: dy = 1; break;
        case 4: dx = -1; break;
        case 8: dx = 1; break;
        default: break;
    }
    const puzzlescript::MaskOffset rowObjectOffset = rowIndex < rule.cellRowMasksCount
        ? game.cellRowMaskOffsets[rule.cellRowMasksFirst + rowIndex]
        : rule.ruleMask;
    const puzzlescript::MaskOffset rowMovementOffset = rowIndex < rule.cellRowMasksMovementsCount
        ? game.cellRowMaskMovementsOffsets[rule.cellRowMasksMovementsFirst + rowIndex]
        : puzzlescript::kNullMaskOffset;

    emitInlineMaskBitsSetCheck(out, game, rowObjectOffset, game.wordCount, "session.scratch.boardMask.data()");
    if (rowMovementOffset != puzzlescript::kNullMaskOffset) {
        emitInlineMaskBitsSetCheck(out, game, rowMovementOffset, game.movementWordCount, "session.scratch.boardMovementMask.data()");
    }
    if (useSessionScratch) {
        out << "    std::vector<int32_t>& " << matchesName << " = session.scratch.singleRowMatchScratch;\n"
            << "    " << matchesName << ".clear();\n";
    } else {
        out << "    std::vector<int32_t> " << matchesName << ";\n";
    }
    out << "    int32_t xmin_" << rowIndex << " = 0;\n"
        << "    int32_t xmax_" << rowIndex << " = currentLevelWidth(session);\n"
        << "    int32_t ymin_" << rowIndex << " = 0;\n"
        << "    int32_t ymax_" << rowIndex << " = currentLevelHeight(session);\n";
    switch (rule.direction) {
        case 1:
            out << "    ymin_" << rowIndex << " += " << (len - 1) << ";\n";
            break;
        case 2:
            out << "    ymax_" << rowIndex << " -= " << (len - 1) << ";\n";
            break;
        case 4:
            out << "    xmin_" << rowIndex << " += " << (len - 1) << ";\n";
            break;
        case 8:
            out << "    xmax_" << rowIndex << " -= " << (len - 1) << ";\n";
            break;
        default:
            out << "    return false;\n";
            break;
    }
    struct AnchorCandidate {
        size_t patternIndex = 0;
        const std::vector<int32_t>* objectIds = nullptr;
    };
    std::vector<AnchorCandidate> anchorCandidates;
    for (size_t patternIndex = 0; patternIndex < row.size(); ++patternIndex) {
        const auto& pattern = row[patternIndex];
        if (!pattern.objectAnchorIds.empty()) {
            anchorCandidates.push_back(AnchorCandidate{patternIndex, &pattern.objectAnchorIds});
        }
        for (const auto& anyIds : pattern.anyObjectAnchorIds) {
            if (!anyIds.empty()) {
                anchorCandidates.push_back(AnchorCandidate{patternIndex, &anyIds});
            }
        }
    }
    if (!anchorCandidates.empty()) {
        out << "    const int32_t validStartCount_" << rowIndex
            << " = std::max(0, xmax_" << rowIndex << " - xmin_" << rowIndex
            << ") * std::max(0, ymax_" << rowIndex << " - ymin_" << rowIndex << ");\n"
            << "    int32_t bestAnchor_" << rowIndex << " = -1;\n"
            << "    uint64_t bestAnchorCount_" << rowIndex << " = 0;\n"
            << "    auto considerAnchor_" << rowIndex << " = [&](int32_t candidateIndex, const int32_t* objectIds, size_t objectIdCount) {\n"
            << "        uint64_t count = 0;\n"
            << "        for (size_t objectIdIndex = 0; objectIdIndex < objectIdCount; ++objectIdIndex) {\n"
            << "            const int32_t objectId = objectIds[objectIdIndex];\n"
            << "            if (objectId >= 0 && objectId < game.objectCount && static_cast<size_t>(objectId) < session.scratch.objectCellCounts.size()) {\n"
            << "                count += session.scratch.objectCellCounts[static_cast<size_t>(objectId)];\n"
            << "            }\n"
            << "        }\n"
            << "        if (count > 0 && (bestAnchor_" << rowIndex << " < 0 || count < bestAnchorCount_" << rowIndex << ")) {\n"
            << "            bestAnchor_" << rowIndex << " = candidateIndex;\n"
            << "            bestAnchorCount_" << rowIndex << " = count;\n"
            << "        }\n"
            << "    };\n";
        for (size_t candidateIndex = 0; candidateIndex < anchorCandidates.size(); ++candidateIndex) {
            const auto& candidate = anchorCandidates[candidateIndex];
            out << "    static constexpr std::array<int32_t, " << candidate.objectIds->size() << "> anchorIds_"
                << rowIndex << "_" << candidateIndex << " = {";
            for (size_t objectIndex = 0; objectIndex < candidate.objectIds->size(); ++objectIndex) {
                if (objectIndex > 0) {
                    out << ", ";
                }
                out << (*candidate.objectIds)[objectIndex];
            }
            out << "};\n";
        }
        out << "    if (!session.scratch.objectCellIndexDirty && !session.scratch.objectCellBits.empty() && validStartCount_" << rowIndex << " > 0) {\n";
        for (size_t candidateIndex = 0; candidateIndex < anchorCandidates.size(); ++candidateIndex) {
            out
                << "        considerAnchor_" << rowIndex << "(" << candidateIndex << ", anchorIds_"
                << rowIndex << "_" << candidateIndex << ".data(), anchorIds_" << rowIndex << "_" << candidateIndex << ".size());\n";
        }
        out << "    }\n"
            << "    if (bestAnchor_" << rowIndex << " >= 0 && bestAnchorCount_" << rowIndex
            << " < static_cast<uint64_t>(std::max(8, validStartCount_" << rowIndex << "))) {\n"
            << "        const int32_t tileCount = currentLevelWidth(session) * currentLevelHeight(session);\n"
            << "        const size_t cellWordCount = static_cast<size_t>((tileCount + 63) / 64);\n"
            << "        auto scanAnchor_" << rowIndex << " = [&](int32_t patternIndex, const int32_t* objectIds, size_t objectIdCount) {\n"
            << "            for (size_t objectIdIndex = 0; objectIdIndex < objectIdCount; ++objectIdIndex) {\n"
            << "                const int32_t objectId = objectIds[objectIdIndex];\n"
            << "                if (objectId < 0 || objectId >= game.objectCount) continue;\n"
            << "                const size_t objectBase = static_cast<size_t>(objectId) * cellWordCount;\n"
            << "                if (objectBase + cellWordCount > session.scratch.objectCellBits.size()) continue;\n"
            << "                for (size_t wordIndex = 0; wordIndex < cellWordCount; ++wordIndex) {\n"
            << "                    uint64_t bits = session.scratch.objectCellBits[objectBase + wordIndex];\n"
            << "                    while (bits != 0) {\n"
            << "                        const int32_t bit = __builtin_ctzll(bits);\n"
            << "                        const int32_t anchorTile = static_cast<int32_t>(wordIndex * 64 + static_cast<size_t>(bit));\n"
            << "                        bits &= bits - 1;\n"
            << "                        if (anchorTile >= tileCount) continue;\n"
            << "                        const int32_t anchorX = anchorTile / currentLevelHeight(session);\n"
            << "                        const int32_t anchorY = anchorTile % currentLevelHeight(session);\n"
            << "                        const int32_t startX = anchorX - patternIndex * (" << dx << ");\n"
            << "                        const int32_t startY = anchorY - patternIndex * (" << dy << ");\n"
            << "                        if (startX < xmin_" << rowIndex << " || startX >= xmax_" << rowIndex
            << " || startY < ymin_" << rowIndex << " || startY >= ymax_" << rowIndex << ") continue;\n";
        if (horizontal) {
            out << "                        const MaskWord* lineObjects = session.scratch.rowMasks.data() + static_cast<size_t>(startY * " << game.wordCount << "U);\n";
            if (rowMovementOffset != puzzlescript::kNullMaskOffset) {
                out << "                        const MaskWord* lineMovements = session.scratch.rowMovementMasks.data() + static_cast<size_t>(startY * " << game.movementWordCount << "U);\n";
            }
        } else {
            out << "                        const MaskWord* lineObjects = session.scratch.columnMasks.data() + static_cast<size_t>(startX * " << game.wordCount << "U);\n";
            if (rowMovementOffset != puzzlescript::kNullMaskOffset) {
                out << "                        const MaskWord* lineMovements = session.scratch.columnMovementMasks.data() + static_cast<size_t>(startX * " << game.movementWordCount << "U);\n";
            }
        }
        emitInlineMaskBitsSetCheck(out, game, rowObjectOffset, game.wordCount, "lineObjects", "                        ", "continue");
        if (rowMovementOffset != puzzlescript::kNullMaskOffset) {
            emitInlineMaskBitsSetCheck(out, game, rowMovementOffset, game.movementWordCount, "lineMovements", "                        ", "continue");
        }
        out << "                        const int32_t startIndex = startX * currentLevelHeight(session) + startY;\n"
            << "                        if (match_" << prefix << "_row" << rowIndex << "(session, startIndex)) "
            << matchesName << ".push_back(startIndex);\n"
            << "                    }\n"
            << "                }\n"
            << "            }\n";
        if (horizontal) {
            out << "            if (" << matchesName << ".size() > 1) {\n"
                << "                std::sort(" << matchesName << ".begin(), " << matchesName << ".end(), [&](int32_t lhs, int32_t rhs) {\n"
                << "                    const int32_t lhsX = lhs / currentLevelHeight(session);\n"
                << "                    const int32_t lhsY = lhs % currentLevelHeight(session);\n"
                << "                    const int32_t rhsX = rhs / currentLevelHeight(session);\n"
                << "                    const int32_t rhsY = rhs % currentLevelHeight(session);\n"
                << "                    return lhsY == rhsY ? lhsX < rhsX : lhsY < rhsY;\n"
                << "                });\n"
                << "                if (objectIdCount > 1) {\n"
                << "                    " << matchesName << ".erase(std::unique(" << matchesName << ".begin(), " << matchesName << ".end()), " << matchesName << ".end());\n"
                << "                }\n"
                << "            }\n";
        } else {
            out << "            if (objectIdCount > 1 && " << matchesName << ".size() > 1) {\n"
                << "                std::sort(" << matchesName << ".begin(), " << matchesName << ".end());\n"
                << "                " << matchesName << ".erase(std::unique(" << matchesName << ".begin(), " << matchesName << ".end()), " << matchesName << ".end());\n"
                << "            }\n";
        }
        out << "        };\n"
            << "        switch (bestAnchor_" << rowIndex << ") {\n";
        for (size_t candidateIndex = 0; candidateIndex < anchorCandidates.size(); ++candidateIndex) {
            const auto& candidate = anchorCandidates[candidateIndex];
            out << "            case " << candidateIndex << ": scanAnchor_" << rowIndex << "("
                << candidate.patternIndex << ", anchorIds_" << rowIndex << "_" << candidateIndex
                << ".data(), anchorIds_" << rowIndex << "_" << candidateIndex << ".size()); break;\n";
        }
        out << "            default: break;\n"
            << "        }\n"
            << "        goto row_matches_done_" << rowIndex << ";\n"
            << "    }\n";
    }
    if (horizontal) {
        out << "    for (int32_t y = ymin_" << rowIndex << "; y < ymax_" << rowIndex << "; ++y) {\n"
            << "        const MaskWord* lineObjects = session.scratch.rowMasks.data() + static_cast<size_t>(y * " << game.wordCount << "U);\n";
        emitInlineMaskBitsSetCheck(out, game, rowObjectOffset, game.wordCount, "lineObjects", "        ", "continue");
        if (rowMovementOffset != puzzlescript::kNullMaskOffset) {
            out << "        const MaskWord* lineMovements = session.scratch.rowMovementMasks.data() + static_cast<size_t>(y * " << game.movementWordCount << "U);\n";
            emitInlineMaskBitsSetCheck(out, game, rowMovementOffset, game.movementWordCount, "lineMovements", "        ", "continue");
        }
        out << "        for (int32_t x = xmin_" << rowIndex << "; x < xmax_" << rowIndex << "; ++x) {\n"
            << "            const int32_t startIndex = x * currentLevelHeight(session) + y;\n"
            << "            if (match_" << prefix << "_row" << rowIndex << "(session, startIndex)) " << matchesName << ".push_back(startIndex);\n"
            << "        }\n"
            << "    }\n";
    } else {
        out << "    for (int32_t x = xmin_" << rowIndex << "; x < xmax_" << rowIndex << "; ++x) {\n"
            << "        const MaskWord* lineObjects = session.scratch.columnMasks.data() + static_cast<size_t>(x * " << game.wordCount << "U);\n";
        emitInlineMaskBitsSetCheck(out, game, rowObjectOffset, game.wordCount, "lineObjects", "        ", "continue");
        if (rowMovementOffset != puzzlescript::kNullMaskOffset) {
            out << "        const MaskWord* lineMovements = session.scratch.columnMovementMasks.data() + static_cast<size_t>(x * " << game.movementWordCount << "U);\n";
            emitInlineMaskBitsSetCheck(out, game, rowMovementOffset, game.movementWordCount, "lineMovements", "        ", "continue");
        }
        out << "        for (int32_t y = ymin_" << rowIndex << "; y < ymax_" << rowIndex << "; ++y) {\n"
            << "            const int32_t startIndex = x * currentLevelHeight(session) + y;\n"
            << "            if (match_" << prefix << "_row" << rowIndex << "(session, startIndex)) " << matchesName << ".push_back(startIndex);\n"
            << "        }\n"
            << "    }\n";
    }
    out << "row_matches_done_" << rowIndex << ":\n"
        << "    if (" << matchesName << ".empty()) return false;\n";
}

void emitCompiledRuleCommandQueue(
    std::ostream& out,
    const puzzlescript::Rule& rule,
    std::string_view ruleExpression
) {
    if (rule.commands.empty()) {
        return;
    }
    if (!canGenerateCompiledRuleCommandQueue(rule)) {
        out << "    compiledRuleQueueCommands(" << ruleExpression << ", commands);\n";
        return;
    }
    const bool currentRuleCancel = std::any_of(rule.commands.begin(), rule.commands.end(), [](const puzzlescript::RuleCommand& command) {
        return command.name == "cancel";
    });
    const bool currentRuleRestart = std::any_of(rule.commands.begin(), rule.commands.end(), [](const puzzlescript::RuleCommand& command) {
        return command.name == "restart";
    });
    out << "    if (compiledRulePrepareCommandQueue(commands, "
        << (currentRuleCancel ? "true" : "false") << ", "
        << (currentRuleRestart ? "true" : "false") << ")) {\n";
    for (const auto& command : rule.commands) {
        const auto kind = compiledRuleCommandKindExpression(command.name);
        if (!kind.has_value()) {
            continue;
        }
        out << "        compiledRuleQueueKnownCommand(commands, "
            << *kind << ", "
            << cppStringLiteral(command.name) << ", "
            << cppStringLiteral(command.argument.value_or("")) << ");\n";
    }
    out << "    }\n";
}

void emitRuleFunctions(
    std::ostream& out,
    const puzzlescript::Game& game,
    const puzzlescript::Rule& rule,
    size_t sourceIndex,
    bool late,
    size_t groupIndex,
    size_t ruleIndex
) {
    const std::string prefix = "s" + std::to_string(sourceIndex)
        + (late ? "_l" : "_e")
        + "_g" + std::to_string(groupIndex)
        + "_r" + std::to_string(ruleIndex);
    if (ruleUsesRuntimeRowHelpers(rule)) {
        out << "bool apply_rule_" << prefix << "(FullState& session, CommandState& commands) {\n"
            << "    const Game& game = *session.game;\n"
            << "    const Rule& rule = game." << (late ? "lateRules" : "rules")
            << "[" << groupIndex << "][" << ruleIndex << "];\n"
            << "    bool changed = false;\n";
        for (size_t rowIndex = 0; rowIndex < rule.patterns.size(); ++rowIndex) {
            out << "    std::vector<CompiledRuleRowMatch> matches_" << rowIndex << ";\n"
                << "    compiledRuleCollectRowMatches(session, rule, " << rowIndex << "U, matches_" << rowIndex << ");\n"
                << "    if (matches_" << rowIndex << ".empty()) return false;\n";
        }
        if (rule.patterns.size() == 1) {
            out << "    for (size_t matchIndex = 0; matchIndex < matches_0.size(); ++matchIndex) {\n"
                << "        const CompiledRuleRowMatch& match = matches_0[matchIndex];\n"
                << "        if (matchIndex > 0 && !compiledRuleRowMatchStillMatches(session, rule, 0U, match)) continue;\n"
                << "        changed = compiledRuleApplyRowMatch(session, rule, 0U, match) || changed;\n"
                << "    }\n";
        } else {
            out << "    std::array<const std::vector<CompiledRuleRowMatch>*, " << rule.patterns.size() << "> allMatches = {";
            for (size_t rowIndex = 0; rowIndex < rule.patterns.size(); ++rowIndex) {
                if (rowIndex > 0) {
                    out << ", ";
                }
                out << "&matches_" << rowIndex;
            }
            out << "};\n"
                << "    std::array<size_t, " << rule.patterns.size() << "> matchIndices{};\n"
                << "    bool firstTuple = true;\n"
                << "    bool done = false;\n"
                << "    while (!done) {\n"
                << "        bool stillMatches = true;\n"
                << "        if (!firstTuple) {\n";
            for (size_t rowIndex = 0; rowIndex < rule.patterns.size(); ++rowIndex) {
                out << "            stillMatches = stillMatches && compiledRuleRowMatchStillMatches(session, rule, "
                    << rowIndex << "U, (*allMatches[" << rowIndex << "])[matchIndices[" << rowIndex << "]]);\n";
            }
            out << "        }\n"
                << "        if (stillMatches) {\n";
            for (size_t rowIndex = 0; rowIndex < rule.patterns.size(); ++rowIndex) {
                out << "            changed = compiledRuleApplyRowMatch(session, rule, " << rowIndex
                    << "U, (*allMatches[" << rowIndex << "])[matchIndices[" << rowIndex << "]]) || changed;\n";
            }
            out << "        }\n"
                << "        firstTuple = false;\n"
                << "        for (size_t carry = 0; carry < matchIndices.size(); ++carry) {\n"
                << "            ++matchIndices[carry];\n"
                << "            if (matchIndices[carry] < allMatches[carry]->size()) break;\n"
                << "            matchIndices[carry] = 0;\n"
                << "            if (carry + 1 == matchIndices.size()) done = true;\n"
                << "        }\n"
                << "    }\n";
        }
        emitCompiledRuleCommandQueue(out, rule, "rule");
        out << "    return changed;\n"
            << "}\n\n";
        return;
    }
    for (size_t rowIndex = 0; rowIndex < rule.patterns.size(); ++rowIndex) {
        emitRuleRowFunctions(out, game, rule, prefix, rowIndex);
    }

    out << "bool apply_rule_" << prefix << "(FullState& session, CommandState& commands) {\n"
        << "    const Game& game = *session.game;\n"
        << "    bool changed = false;\n";
    const bool oneRow = rule.patterns.size() == 1;
    for (size_t rowIndex = 0; rowIndex < rule.patterns.size(); ++rowIndex) {
        emitCollectRowMatches(out, game, rule, prefix, rowIndex, "matches_" + std::to_string(rowIndex), oneRow);
    }
    if (oneRow) {
        out << "    for (size_t matchIndex = 0; matchIndex < matches_0.size(); ++matchIndex) {\n"
            << "        const int32_t startIndex = matches_0[matchIndex];\n"
            << "        if (matchIndex > 0 && !match_" << prefix << "_row0(session, startIndex)) continue;\n"
            << "        changed = apply_replacements_" << prefix << "_row0(session, startIndex) || changed;\n"
            << "    }\n";
    } else {
        out << "    std::array<const std::vector<int32_t>*, " << rule.patterns.size() << "> allMatches = {";
        for (size_t rowIndex = 0; rowIndex < rule.patterns.size(); ++rowIndex) {
            if (rowIndex > 0) {
                out << ", ";
            }
            out << "&matches_" << rowIndex;
        }
        out << "};\n"
            << "    std::array<size_t, " << rule.patterns.size() << "> matchIndices{};\n"
            << "    bool firstTuple = true;\n"
            << "    bool done = false;\n"
            << "    while (!done) {\n"
            << "        bool stillMatches = true;\n"
            << "        if (!firstTuple) {\n";
        for (size_t rowIndex = 0; rowIndex < rule.patterns.size(); ++rowIndex) {
            out << "            stillMatches = stillMatches && match_" << prefix << "_row" << rowIndex
                << "(session, (*allMatches[" << rowIndex << "])[matchIndices[" << rowIndex << "]]);\n";
        }
        out << "        }\n"
            << "        if (stillMatches) {\n";
        for (size_t rowIndex = 0; rowIndex < rule.patterns.size(); ++rowIndex) {
            out << "            changed = apply_replacements_" << prefix << "_row" << rowIndex
                << "(session, (*allMatches[" << rowIndex << "])[matchIndices[" << rowIndex << "]]) || changed;\n";
        }
        out << "        }\n"
            << "        firstTuple = false;\n"
            << "        for (size_t carry = 0; carry < matchIndices.size(); ++carry) {\n"
            << "            ++matchIndices[carry];\n"
            << "            if (matchIndices[carry] < allMatches[carry]->size()) break;\n"
            << "            matchIndices[carry] = 0;\n"
            << "            if (carry + 1 == matchIndices.size()) done = true;\n"
            << "        }\n"
            << "    }\n";
    }
    emitCompiledRuleCommandQueue(
        out,
        rule,
        std::string("game.")
            + (late ? "lateRules" : "rules")
            + "[" + std::to_string(groupIndex) + "][" + std::to_string(ruleIndex) + "]"
    );
    out << "    return changed;\n"
        << "}\n\n";
}

struct CodegenSource {
    std::filesystem::path path;
    std::string source;
    std::shared_ptr<const puzzlescript::Game> game;
    uint64_t hash = 0;
};

struct CompiledRulesCoverage {
    uint64_t sources = 0;
    uint64_t earlyGroups = 0;
    uint64_t earlyRules = 0;
    uint64_t lateGroups = 0;
    uint64_t lateRules = 0;
    uint64_t compiledGroups = 0;
    uint64_t compiledRules = 0;
    uint64_t compiledEarlyGroups = 0;
    uint64_t compiledEarlyRules = 0;
    uint64_t compiledLateGroups = 0;
    uint64_t compiledLateRules = 0;
    std::unordered_map<std::string, uint64_t> missedGroupsByReason;
};

CompiledRulesCoverage measureCompiledRulesCoverage(
    const std::vector<CodegenSource>& sources,
    const CompiledRulesOptions& options
) {
    CompiledRulesCoverage coverage;
    coverage.sources = sources.size();
    for (const auto& source : sources) {
        const auto& game = *source.game;
        for (const auto& group : game.rules) {
            ++coverage.earlyGroups;
            coverage.earlyRules += group.size();
            const std::string reason = compiledGroupMissReason(group, options);
            if (reason.empty()) {
                ++coverage.compiledGroups;
                coverage.compiledRules += group.size();
                ++coverage.compiledEarlyGroups;
                coverage.compiledEarlyRules += group.size();
            } else {
                ++coverage.missedGroupsByReason[reason];
            }
        }
        for (const auto& group : game.lateRules) {
            ++coverage.lateGroups;
            coverage.lateRules += group.size();
            const std::string reason = compiledGroupMissReason(group, options);
            if (reason.empty()) {
                ++coverage.compiledGroups;
                coverage.compiledRules += group.size();
                ++coverage.compiledLateGroups;
                coverage.compiledLateRules += group.size();
            } else {
                ++coverage.missedGroupsByReason[reason];
            }
        }
    }
    return coverage;
}

void printCompiledRulesCoverage(const CompiledRulesCoverage& coverage) {
    std::cerr << "compiled-rules-coverage:"
              << " sources=" << coverage.sources
              << " early_groups=" << coverage.earlyGroups
              << " early_rules=" << coverage.earlyRules
              << " compiled_early_groups=" << coverage.compiledEarlyGroups
              << " compiled_early_rules=" << coverage.compiledEarlyRules
              << " compiled_groups=" << coverage.compiledGroups
              << " compiled_rules=" << coverage.compiledRules
              << " late_groups=" << coverage.lateGroups
              << " late_rules=" << coverage.lateRules
              << " compiled_late_groups=" << coverage.compiledLateGroups
              << " compiled_late_rules=" << coverage.compiledLateRules
              << "\n";

    static const std::array<std::string_view, 10> kReasonOrder = {
        "random_group",
        "random_rule",
        "rigid",
        "row_limit",
        "ellipsis",
        "missing_ellipsis_metadata",
        "empty_row",
        "non_cell_pattern",
        "random_replacement",
        "empty_group",
    };
    std::cerr << "compiled-rules-misses:";
    bool printedAny = false;
    for (const std::string_view reason : kReasonOrder) {
        const auto it = coverage.missedGroupsByReason.find(std::string(reason));
        if (it == coverage.missedGroupsByReason.end() || it->second == 0) {
            continue;
        }
        std::cerr << " " << reason << "=" << it->second;
        printedAny = true;
    }
    if (!printedAny) {
        std::cerr << " none=0";
    }
    std::cerr << "\n";
}

std::vector<std::string> orderedCompiledRuleMissReasons(const CompiledRulesCoverage& coverage) {
    static const std::array<std::string_view, 10> kReasonOrder = {
        "random_group",
        "random_rule",
        "rigid",
        "row_limit",
        "ellipsis",
        "missing_ellipsis_metadata",
        "empty_row",
        "non_cell_pattern",
        "random_replacement",
        "empty_group",
    };
    std::vector<std::string> reasons;
    std::set<std::string> seen;
    for (const std::string_view reason : kReasonOrder) {
        const std::string key(reason);
        const auto it = coverage.missedGroupsByReason.find(key);
        if (it != coverage.missedGroupsByReason.end() && it->second != 0) {
            reasons.push_back(key);
            seen.insert(key);
        }
    }
    for (const auto& [reason, count] : coverage.missedGroupsByReason) {
        if (count != 0 && seen.insert(reason).second) {
            reasons.push_back(reason);
        }
    }
    return reasons;
}

void appendCompiledRulesCoverageJsonFields(std::ostream& out, const CompiledRulesCoverage& coverage) {
    out << "\"sources\":" << coverage.sources
        << ",\"early_groups\":" << coverage.earlyGroups
        << ",\"early_rules\":" << coverage.earlyRules
        << ",\"compiled_early_groups\":" << coverage.compiledEarlyGroups
        << ",\"compiled_early_rules\":" << coverage.compiledEarlyRules
        << ",\"late_groups\":" << coverage.lateGroups
        << ",\"late_rules\":" << coverage.lateRules
        << ",\"compiled_late_groups\":" << coverage.compiledLateGroups
        << ",\"compiled_late_rules\":" << coverage.compiledLateRules
        << ",\"compiled_groups\":" << coverage.compiledGroups
        << ",\"compiled_rules\":" << coverage.compiledRules
        << ",\"fully_compiled\":" << (coverage.missedGroupsByReason.empty() ? "true" : "false")
        << ",\"misses\":{";
    bool first = true;
    for (const std::string& reason : orderedCompiledRuleMissReasons(coverage)) {
        const auto it = coverage.missedGroupsByReason.find(reason);
        if (it == coverage.missedGroupsByReason.end() || it->second == 0) {
            continue;
        }
        if (!first) {
            out << ",";
        }
        first = false;
        out << jsonStringLiteral(reason) << ":" << it->second;
    }
    out << "}";
}

void appendJsonCountObject(
    std::ostream& out,
    const std::unordered_map<std::string, size_t>& counts,
    const std::vector<std::string_view>& preferredOrder
) {
    out << "{";
    bool first = true;
    for (const std::string_view key : preferredOrder) {
        const auto it = counts.find(std::string(key));
        if (it == counts.end() || it->second == 0) {
            continue;
        }
        if (!first) {
            out << ",";
        }
        first = false;
        out << jsonStringLiteral(key) << ":" << it->second;
    }
    for (const auto& [key, count] : counts) {
        if (count == 0) {
            continue;
        }
        if (std::find(preferredOrder.begin(), preferredOrder.end(), std::string_view(key)) != preferredOrder.end()) {
            continue;
        }
        if (!first) {
            out << ",";
        }
        first = false;
        out << jsonStringLiteral(key) << ":" << count;
    }
    out << "}";
}

void appendSpecializedFullTurnAggregateJsonFields(
    std::ostream& out,
    size_t sourceCount,
    size_t earlyRuleLoopsGenerated,
    size_t lateRuleLoopsGenerated,
    size_t commandNone,
    size_t commandGeneratedQueue,
    size_t commandUnknown,
    size_t wholeTurnSupported,
    const std::unordered_map<std::string, size_t>& wholeTurnFallbackReasons
) {
    auto appendObject = [&](std::string_view key) {
        out << jsonStringLiteral(key) << ":{"
            << "\"sources\":" << sourceCount
            << ",\"backend_codegen_available\":" << sourceCount
            << ",\"early_rule_loops_generated\":" << earlyRuleLoopsGenerated
            << ",\"late_rule_loops_generated\":" << lateRuleLoopsGenerated
            << ",\"fully_generated\":" << wholeTurnSupported
            << ",\"whole_turn_supported\":" << wholeTurnSupported
            << ",\"command_status_counts\":{"
            << "\"none\":" << commandNone
            << ",\"generated_queue_interpreter_tail\":" << commandGeneratedQueue
            << ",\"unknown_interpreter\":" << commandUnknown
            << "}"
            << ",\"whole_turn_fallback_reason_counts\":";
        appendJsonCountObject(
            out,
            wholeTurnFallbackReasons,
            {"early_rule_loops_interpreter", "late_rule_loops_interpreter", "unsupported_command", "movement_interpreter", "interpreter_delegation"}
        );
        out << ",\"misses\":{";
        if (sourceCount > wholeTurnSupported) {
            out << "\"interpreter_delegation\":" << (sourceCount - wholeTurnSupported);
        }
        out << "}}";
    };
    appendObject("specialized_full_turn");
    out << ",";
    appendObject("compiled_tick");
}

void appendSpecializedFullTurnSourceJsonFields(
    std::ostream& out,
    const SpecializedFullTurnSupport& support
) {
    auto appendObject = [&](std::string_view key) {
        out << jsonStringLiteral(key) << ":{"
            << "\"backend_codegen_available\":true"
            << ",\"step_entry\":true"
            << ",\"tick_entry\":true"
            << ",\"fully_generated\":" << (support.wholeTurnSupported ? "true" : "false")
            << ",\"whole_turn_supported\":" << (support.wholeTurnSupported ? "true" : "false")
            << ",\"whole_turn_fallback_reason\":" << jsonStringLiteral(support.wholeTurnFallbackReason)
            << ",\"features\":{"
            << "\"rule_loops\":" << jsonStringLiteral(
                support.earlyRuleLoopsGenerated && support.lateRuleLoopsGenerated
                    ? "early_late_generated"
                    : (support.earlyRuleLoopsGenerated ? "early_generated_late_interpreter" : "interpreter")
            )
            << ",\"commands\":" << jsonStringLiteral(support.commandStatus)
            << ",\"movement\":\"interpreter\""
            << ",\"win_conditions\":\"interpreter\""
            << ",\"level_transitions\":\"interpreter\""
            << ",\"state_layout\":\"session\""
            << "}"
            << ",\"misses\":{\"interpreter_delegation\":1}"
            << "}";
    };
    appendObject("specialized_full_turn");
    out << ",";
    appendObject("compiled_tick");
}

void appendCompactTurnAggregateJsonFields(
    std::ostream& out,
    size_t sourceCount,
    size_t supported,
    size_t nativeKernelSupported,
    size_t interpreterBridgeSupported,
    const std::unordered_map<std::string, size_t>& fallbackReasons,
    const std::unordered_map<std::string, size_t>& nativeFallbackReasons
) {
    const std::vector<std::string_view> compactFallbackReasonOrder{
        "supported",
        "interpreter_bridge",
        "native_compact_generator_rebuild",
    };
    out << "\"compact_turn\":{"
        << "\"sources\":" << sourceCount
        << ",\"backend_codegen_available\":" << sourceCount
        << ",\"whole_turn_supported\":" << supported
        << ",\"native_kernel_supported\":" << nativeKernelSupported
        << ",\"interpreter_bridge_supported\":" << interpreterBridgeSupported
        << ",\"whole_turn_fallback_reason_counts\":";
    appendJsonCountObject(out, fallbackReasons, compactFallbackReasonOrder);
    out << ",\"native_kernel_fallback_reason_counts\":";
    appendJsonCountObject(out, nativeFallbackReasons, compactFallbackReasonOrder);
    out << ",\"misses\":{";
    if (sourceCount > supported) {
        out << "\"interpreter_delegation\":" << (sourceCount - supported);
    }
    out << "}}";
}

void appendCompactTurnSourceJsonFields(
    std::ostream& out,
    const CompactTurnSupport& support
) {
    out << "\"compact_turn\":{"
        << "\"backend_codegen_available\":true"
        << ",\"step_entry\":true"
        << ",\"whole_turn_supported\":" << (support.supported ? "true" : "false")
        << ",\"whole_turn_fallback_reason\":" << jsonStringLiteral(support.fallbackReason)
        << ",\"mode\":" << jsonStringLiteral(support.interpreterBridge ? "interpreter_bridge" : "native_kernel")
        << ",\"native_kernel_supported\":" << ((!support.interpreterBridge && support.supported) ? "true" : "false")
        << ",\"native_kernel_fallback_reason\":" << jsonStringLiteral(support.nativeFallbackReason)
        << ",\"features\":{"
        << "\"state_layout\":\"compact_object_bits\""
        << ",\"movement\":" << jsonStringLiteral(support.interpreterBridge ? "interpreter_bridge" : "native_generator")
        << ",\"rules\":" << jsonStringLiteral(support.interpreterBridge ? "interpreter_bridge" : "native_generator")
        << ",\"late_rules\":" << jsonStringLiteral(support.interpreterBridge ? "interpreter_bridge" : "native_generator")
        << ",\"win_conditions\":" << jsonStringLiteral(support.interpreterBridge ? "interpreter_bridge" : "native_generator")
        << "}"
        << ",\"misses\":{";
    if (!support.supported) {
        out << jsonStringLiteral(support.fallbackReason) << ":1";
    }
    out << "}}";
}

std::string generateCompiledRulesCoverageJson(
    const std::vector<CodegenSource>& sources,
    const CompiledRulesOptions& options,
    const CompiledRulesCoverage& aggregateCoverage
) {
    std::ostringstream out;
    size_t earlyRuleLoopsGenerated = 0;
    size_t lateRuleLoopsGenerated = 0;
    size_t commandNone = 0;
    size_t commandGeneratedQueue = 0;
    size_t commandUnknown = 0;
    size_t wholeTurnSupported = 0;
    std::unordered_map<std::string, size_t> wholeTurnFallbackReasons;
    size_t compactTurnSupported = 0;
    size_t compactTurnNativeSupported = 0;
    size_t compactTurnInterpreterBridgeSupported = 0;
    std::unordered_map<std::string, size_t> compactTurnFallbackReasons;
    std::unordered_map<std::string, size_t> compactTurnNativeFallbackReasons;
    for (const CodegenSource& source : sources) {
        const SpecializedFullTurnSupport support = source.game
            ? specializedFullTurnSupportForGame(*source.game, options)
            : specializedFullTurnSupportForMissingGame();
        if (support.earlyRuleLoopsGenerated) {
            ++earlyRuleLoopsGenerated;
        }
        if (support.lateRuleLoopsGenerated) {
            ++lateRuleLoopsGenerated;
        }
        if (support.commandStatus == "none") {
            ++commandNone;
        } else if (support.commandStatus == "generated_queue_interpreter_tail") {
            ++commandGeneratedQueue;
        } else {
            ++commandUnknown;
        }
        if (support.wholeTurnSupported) {
            ++wholeTurnSupported;
        } else {
            ++wholeTurnFallbackReasons[support.wholeTurnFallbackReason];
        }
        const CompactTurnSupport compactSupport = source.game
            ? compactTurnSupportForGame(*source.game)
            : CompactTurnSupport{};
        if (compactSupport.supported) {
            ++compactTurnSupported;
        }
        if (compactSupport.interpreterBridge) {
            ++compactTurnInterpreterBridgeSupported;
        } else if (compactSupport.supported) {
            ++compactTurnNativeSupported;
        }
        ++compactTurnFallbackReasons[compactSupport.fallbackReason];
        ++compactTurnNativeFallbackReasons[compactSupport.nativeFallbackReason];
    }
    out << "{\n"
        << "  \"max_rows\":" << options.maxRows << ",\n"
        << "  \"aggregate\":{";
    appendCompiledRulesCoverageJsonFields(out, aggregateCoverage);
    out << ",";
    appendSpecializedFullTurnAggregateJsonFields(
        out,
        sources.size(),
        earlyRuleLoopsGenerated,
        lateRuleLoopsGenerated,
        commandNone,
        commandGeneratedQueue,
        commandUnknown,
        wholeTurnSupported,
        wholeTurnFallbackReasons
    );
    out << ",";
    appendCompactTurnAggregateJsonFields(
        out,
        sources.size(),
        compactTurnSupported,
        compactTurnNativeSupported,
        compactTurnInterpreterBridgeSupported,
        compactTurnFallbackReasons,
        compactTurnNativeFallbackReasons
    );
    out << "},\n"
        << "  \"sources\":[\n";
    for (size_t index = 0; index < sources.size(); ++index) {
        const CompiledRulesCoverage sourceCoverage = measureCompiledRulesCoverage(
            std::vector<CodegenSource>{sources[index]},
            options
        );
        out << "    {\"index\":" << index
            << ",\"path\":" << jsonStringLiteral(sources[index].path.string())
            << ",\"source_hash\":" << sources[index].hash
            << ",";
        appendCompiledRulesCoverageJsonFields(out, sourceCoverage);
        out << ",";
        appendSpecializedFullTurnSourceJsonFields(
            out,
            sources[index].game
                ? specializedFullTurnSupportForGame(*sources[index].game, options)
                : specializedFullTurnSupportForMissingGame()
        );
        out << ",";
        appendCompactTurnSourceJsonFields(
            out,
            sources[index].game
                ? compactTurnSupportForGame(*sources[index].game)
                : CompactTurnSupport{}
        );
        out << "}";
        if (index + 1 < sources.size()) {
            out << ",";
        }
        out << "\n";
    }
    out << "  ]\n"
        << "}\n";
    return out.str();
}

CodegenSource compileCodegenSource(const std::filesystem::path& path) {
    CodegenSource result;
    result.path = path;
    result.source = readFile(path);
    if (result.source.empty() || result.source.back() != '\n') {
        result.source.push_back('\n');
    }
    result.hash = puzzlescript::compiledRulesHashSource(result.source);
    puzzlescript::compiler::DiagnosticSink diagnostics;
    const auto parserState = puzzlescript::compiler::parseSource(result.source, diagnostics);
    puzzlescript::LoadedGame loadedGame;
    if (auto error = puzzlescript::compiler::lowerToRuntimeGame(parserState, loadedGame)) {
        throw std::runtime_error(path.string() + ": " + error->message);
    }
    result.game = std::move(loadedGame.information);
    if (!result.game) {
        throw std::runtime_error(path.string() + ": lowering produced no runtime game");
    }
    return result;
}

CodegenSource compileCodegenSourceText(std::string label, std::string source) {
    CodegenSource result;
    result.path = std::move(label);
    result.source = std::move(source);
    if (result.source.empty() || result.source.back() != '\n') {
        result.source.push_back('\n');
    }
    result.hash = puzzlescript::compiledRulesHashSource(result.source);
    puzzlescript::compiler::DiagnosticSink diagnostics;
    const auto parserState = puzzlescript::compiler::parseSource(result.source, diagnostics);
    puzzlescript::LoadedGame loadedGame;
    if (auto error = puzzlescript::compiler::lowerToRuntimeGame(parserState, loadedGame)) {
        throw std::runtime_error(result.path.string() + ": " + error->message);
    }
    result.game = std::move(loadedGame.information);
    if (!result.game) {
        throw std::runtime_error(result.path.string() + ": lowering produced no runtime game");
    }
    return result;
}

std::vector<std::filesystem::path> collectCompiledRuleSourcePaths(const std::filesystem::path& path) {
    std::vector<std::filesystem::path> paths;
    if (std::filesystem::is_directory(path)) {
        for (const auto& entry : std::filesystem::recursive_directory_iterator(path)) {
            if (!entry.is_regular_file() || entry.path().extension() != ".txt") {
                continue;
            }
            paths.push_back(entry.path());
        }
        std::sort(paths.begin(), paths.end());
    } else {
        paths.push_back(path);
    }
    return paths;
}

std::vector<CodegenSource> collectCompiledRuleSources(const std::filesystem::path& path) {
    std::vector<CodegenSource> sources;
    std::set<uint64_t> seenHashes;
    auto addSource = [&](CodegenSource source) {
        if (seenHashes.insert(source.hash).second) {
            sources.push_back(std::move(source));
        }
    };

    if (path.extension() == ".js") {
        const auto cases = parseSimulationCorpusCases(loadJsDataArrayAsJson(path));
        for (const auto& testCase : cases) {
            addSource(compileCodegenSourceText(
                path.string() + "#" + std::to_string(testCase.index + 1) + ":" + testCase.name,
                testCase.source
            ));
        }
        return sources;
    }

    for (const auto& sourcePath : collectCompiledRuleSourcePaths(path)) {
        addSource(compileCodegenSource(sourcePath));
    }
    return sources;
}

std::string generateCompiledRulesCpp(
    const std::vector<CodegenSource>& sources,
    std::string_view symbol,
    const CompiledRulesOptions& options,
    bool emitGlobalFinder = true,
    std::string_view backendAccessorSymbol = {},
    std::string_view tickBackendAccessorSymbol = {},
    std::string_view compactTurnBackendAccessorSymbol = {},
    bool compactTurnOnly = false
) {
    std::ostringstream out;
    const std::string safeSymbol = safeCppIdentifier(symbol);
    out << "// Generated by puzzlescript_cpp specialize-rulegroups. Do not edit by hand.\n";
    if (sources.size() == 1) {
        out << "// Source: " << sources[0].path.string() << "\n"
            << "// Source hash: " << sources[0].hash << "\n";
    }
    out << "#include <algorithm>\n"
        << "#include <array>\n"
        << "#include <cstddef>\n"
        << "#include <cstdint>\n"
        << "#include <vector>\n"
        << "#include \"runtime/compiled_rules.hpp\"\n\n"
        << "namespace {\n"
        << "using namespace puzzlescript;\n\n";

    uint32_t totalCompiledRules = 0;
    uint32_t totalCompiledGroups = 0;
    for (size_t sourceIndex = 0; sourceIndex < sources.size(); ++sourceIndex) {
        const auto& source = sources[sourceIndex];
        const auto& game = *source.game;
        const SpecializedFullTurnSupport tickSupport = specializedFullTurnSupportForGame(game, options);
        const CompactTurnSupport compactTurnSupport = compactTurnSupportForGame(game);
        uint32_t sourceCompiledRules = 0;
        uint32_t sourceCompiledGroups = 0;
        if (!compactTurnOnly) {
            for (size_t groupIndex = 0; groupIndex < game.rules.size(); ++groupIndex) {
                const auto& group = game.rules[groupIndex];
                if (!isCompilableGroup(group, options)) {
                    continue;
                }
                ++sourceCompiledGroups;
                sourceCompiledRules += static_cast<uint32_t>(group.size());
                if (!group[0].isRandom) {
                    for (size_t ruleIndex = 0; ruleIndex < group.size(); ++ruleIndex) {
                        emitRuleFunctions(out, game, group[ruleIndex], sourceIndex, false, groupIndex, ruleIndex);
                    }
                }
            }
            for (size_t groupIndex = 0; groupIndex < game.lateRules.size(); ++groupIndex) {
                const auto& group = game.lateRules[groupIndex];
                if (!isCompilableGroup(group, options)) {
                    continue;
                }
                ++sourceCompiledGroups;
                sourceCompiledRules += static_cast<uint32_t>(group.size());
                if (!group[0].isRandom) {
                    for (size_t ruleIndex = 0; ruleIndex < group.size(); ++ruleIndex) {
                        emitRuleFunctions(out, game, group[ruleIndex], sourceIndex, true, groupIndex, ruleIndex);
                    }
                }
            }
        }
        totalCompiledRules += sourceCompiledRules;
        totalCompiledGroups += sourceCompiledGroups;

        out << "SpecializedRulegroupOutcome apply_source_" << sourceIndex << "(FullState& session, int32_t groupIndex, bool late, CommandState& commands) {\n"
            << "    const Game& game = *session.game;\n"
            << (compactTurnOnly
                ? "    (void)game;\n    (void)groupIndex;\n    (void)late;\n    (void)commands;\n    return {false, false};\n}\n\n"
                : "")
            ;
        if (!compactTurnOnly) {
            out
            << "    if (late) {\n"
            << "    switch (groupIndex) {\n";
        for (size_t groupIndex = 0; groupIndex < game.lateRules.size(); ++groupIndex) {
            const auto& group = game.lateRules[groupIndex];
            if (!isCompilableGroup(group, options)) {
                continue;
            }
            out << "        case " << groupIndex << ": {\n"
                << (group[0].isRandom
                    ? "            return {true, compiledRuleApplyRandomGroup(session, game.lateRules[groupIndex], commands)};\n"
                    : "")
                << (group[0].isRandom ? "        }\n" : "");
            if (group[0].isRandom) {
                continue;
            }
            out
                << "            bool hasChanges = false;\n"
                << "            bool madeChange = true;\n"
                << "            int loopCount = 0;\n"
                << "            if (session.scratch.anyMasksDirty || session.scratch.objectCellIndexDirty) {\n"
                << "                compiledRuleRebuildMasks(session);\n"
                << "            }\n"
                << "            while (madeChange && loopCount++ < 200) {\n"
                << "                madeChange = false;\n";
            for (size_t ruleIndex = 0; ruleIndex < group.size(); ++ruleIndex) {
                out << "                if (apply_rule_s" << sourceIndex << "_l_g" << groupIndex << "_r" << ruleIndex << "(session, commands)) {\n"
                    << "                    madeChange = true;\n"
                    << "                    if (session.scratch.anyMasksDirty || session.scratch.objectCellIndexDirty) {\n"
                    << "                        compiledRuleRebuildMasks(session);\n"
                    << "                    }\n"
                    << "                }\n";
            }
            out << "                hasChanges = hasChanges || madeChange;\n"
                << "            }\n"
                << "            return {true, hasChanges};\n"
                << "        }\n";
        }
        out << "        default: return {false, false};\n"
            << "    }\n"
            << "    }\n"
            << "    switch (groupIndex) {\n";
        for (size_t groupIndex = 0; groupIndex < game.rules.size(); ++groupIndex) {
            const auto& group = game.rules[groupIndex];
            if (!isCompilableGroup(group, options)) {
                continue;
            }
            out << "        case " << groupIndex << ": {\n"
                << (group[0].isRandom
                    ? "            return {true, compiledRuleApplyRandomGroup(session, game.rules[groupIndex], commands)};\n"
                    : "")
                << (group[0].isRandom ? "        }\n" : "");
            if (group[0].isRandom) {
                continue;
            }
            out
                << "            bool hasChanges = false;\n"
                << "            bool madeChange = true;\n"
                << "            int loopCount = 0;\n"
                << "            if (session.scratch.anyMasksDirty || session.scratch.objectCellIndexDirty) {\n"
                << "                compiledRuleRebuildMasks(session);\n"
                << "            }\n"
                << "            while (madeChange && loopCount++ < 200) {\n"
                << "                madeChange = false;\n";
            for (size_t ruleIndex = 0; ruleIndex < group.size(); ++ruleIndex) {
                out << "                if (apply_rule_s" << sourceIndex << "_e_g" << groupIndex << "_r" << ruleIndex << "(session, commands)) {\n"
                    << "                    madeChange = true;\n"
                    << "                    if (session.scratch.anyMasksDirty || session.scratch.objectCellIndexDirty) {\n"
                    << "                        compiledRuleRebuildMasks(session);\n"
                    << "                    }\n"
                    << "                }\n";
            }
            out << "                hasChanges = hasChanges || madeChange;\n"
                << "            }\n"
                << "            return {true, hasChanges};\n"
                << "        }\n";
        }
        out << "        default: return {false, false};\n"
            << "    }\n"
            << "}\n\n";
        }

        out << "SpecializedRulegroupsForInterpretedTurnOutcome apply_early_groups_source_" << sourceIndex << "(FullState& session, CommandState& commands, std::vector<bool>* bannedGroups) {\n";
        if (compactTurnOnly || !areAllGroupsCompilable(game.rules, options)) {
            out << "    (void)session;\n"
                << "    (void)commands;\n"
                << "    (void)bannedGroups;\n"
                << "    return {false, false};\n"
                << "}\n\n";
        } else {
            out << "    bool loopPropagated = false;\n"
                << "    bool hasChanges = false;\n"
                << "    int32_t loopCount = 0;\n"
                << "    int32_t groupIndex = 0;\n"
                << "    constexpr int32_t groupCount = " << game.rules.size() << ";\n"
                << "    while (groupIndex < groupCount) {\n"
                << "        bool groupChanged = false;\n"
                << "        if (bannedGroups == nullptr\n"
                << "            || static_cast<size_t>(groupIndex) >= bannedGroups->size()\n"
                << "            || !(*bannedGroups)[static_cast<size_t>(groupIndex)]) {\n"
                << "            const SpecializedRulegroupOutcome outcome = apply_source_" << sourceIndex << "(session, groupIndex, false, commands);\n"
                << "            if (!outcome.handled) {\n"
                << "                return {false, false};\n"
                << "            }\n"
                << "            groupChanged = outcome.changed;\n"
                << "        }\n"
                << "        loopPropagated = groupChanged || loopPropagated;\n"
                << "        hasChanges = groupChanged || hasChanges;\n"
                << "        if (loopPropagated) {\n"
                << "            switch (groupIndex) {\n";
            for (size_t loopIndex = 0; loopIndex < game.loopPoint.entries.size(); ++loopIndex) {
                const auto& target = game.loopPoint.entries[loopIndex];
                if (!target.has_value()) {
                    continue;
                }
                out << "                case " << loopIndex << ": {\n"
                    << "                    groupIndex = " << *target << ";\n"
                    << "                    loopPropagated = false;\n"
                    << "                    if (++loopCount > 200) {\n"
                    << "                        return {true, hasChanges};\n"
                    << "                    }\n"
                    << "                    continue;\n"
                    << "                }\n";
            }
            out << "                default: break;\n"
                << "            }\n"
                << "        }\n"
                << "        ++groupIndex;\n"
                << "    }\n"
                << "    return {true, hasChanges};\n"
                << "}\n\n";
        }

        out << "SpecializedRulegroupsForInterpretedTurnOutcome apply_late_groups_source_" << sourceIndex << "(FullState& session, CommandState& commands, std::vector<bool>* bannedGroups) {\n";
        if (compactTurnOnly || !areAllGroupsCompilable(game.lateRules, options)) {
            out << "    (void)session;\n"
                << "    (void)commands;\n"
                << "    (void)bannedGroups;\n"
                << "    return {false, false};\n"
                << "}\n\n";
        } else {
            out << "    bool loopPropagated = false;\n"
                << "    bool hasChanges = false;\n"
                << "    int32_t loopCount = 0;\n"
                << "    int32_t groupIndex = 0;\n"
                << "    constexpr int32_t groupCount = " << game.lateRules.size() << ";\n"
                << "    while (groupIndex < groupCount) {\n"
                << "        bool groupChanged = false;\n"
                << "        if (bannedGroups == nullptr\n"
                << "            || static_cast<size_t>(groupIndex) >= bannedGroups->size()\n"
                << "            || !(*bannedGroups)[static_cast<size_t>(groupIndex)]) {\n"
                << "            const SpecializedRulegroupOutcome outcome = apply_source_" << sourceIndex << "(session, groupIndex, true, commands);\n"
                << "            if (!outcome.handled) {\n"
                << "                return {false, false};\n"
                << "            }\n"
                << "            groupChanged = outcome.changed;\n"
                << "        }\n"
                << "        loopPropagated = groupChanged || loopPropagated;\n"
                << "        hasChanges = groupChanged || hasChanges;\n"
                << "        if (loopPropagated) {\n"
                << "            switch (groupIndex) {\n";
            for (size_t loopIndex = 0; loopIndex < game.lateLoopPoint.entries.size(); ++loopIndex) {
                const auto& target = game.lateLoopPoint.entries[loopIndex];
                if (!target.has_value()) {
                    continue;
                }
                out << "                case " << loopIndex << ": {\n"
                    << "                    groupIndex = " << *target << ";\n"
                    << "                    loopPropagated = false;\n"
                    << "                    if (++loopCount > 200) {\n"
                    << "                        return {true, hasChanges};\n"
                    << "                    }\n"
                    << "                    continue;\n"
                    << "                }\n";
            }
            out << "                default: break;\n"
                << "            }\n"
                << "        }\n"
                << "        ++groupIndex;\n"
                << "    }\n"
                << "    return {true, hasChanges};\n"
                << "}\n\n";
        }

        out << "const SpecializedRulegroupsBackend backend_" << sourceIndex << " = {\n"
            << "    " << source.hash << "ULL,\n"
            << "    " << cppStringLiteral(source.path.string()) << ",\n"
            << "    apply_source_" << sourceIndex << ",\n"
            << "    " << sourceCompiledRules << "U,\n"
            << "    " << sourceCompiledGroups << "U,\n"
            << "};\n\n";

        out << "SpecializedFullTurnOutcome tick_step_source_" << sourceIndex << "(FullState& session, ps_input input, RuntimeStepOptions options) {\n"
            << (compactTurnOnly
                ? "    (void)session;\n    (void)input;\n    (void)options;\n    return {false, {}};\n"
                : "    return {true, puzzlescript::interpretedStepWithSpecializedRulegroups(session, input, options, apply_early_groups_source_" + std::to_string(sourceIndex) + ", apply_late_groups_source_" + std::to_string(sourceIndex) + ")};\n")
            << "}\n\n"
            << "SpecializedFullTurnOutcome tick_source_" << sourceIndex << "(FullState& session, RuntimeStepOptions options) {\n"
            << (compactTurnOnly
                ? "    (void)session;\n    (void)options;\n    return {false, {}};\n"
                : "    return {true, puzzlescript::interpretedTickWithSpecializedRulegroups(session, options, apply_early_groups_source_" + std::to_string(sourceIndex) + ", apply_late_groups_source_" + std::to_string(sourceIndex) + ")};\n")
            << "}\n\n"
            << "const SpecializedFullTurnBackend tick_backend_" << sourceIndex << " = {\n"
            << "    " << source.hash << "ULL,\n"
            << "    " << cppStringLiteral(source.path.string()) << ",\n"
            << "    tick_step_source_" << sourceIndex << ",\n"
            << "    tick_source_" << sourceIndex << ",\n"
            << "    {" << (!compactTurnOnly && tickSupport.wholeTurnSupported ? "true" : "false")
            << ", " << cppStringLiteral(compactTurnOnly ? "compact_turn_only" : tickSupport.wholeTurnFallbackReason) << "},\n"
            << "};\n\n"
            << "SpecializedCompactTurnOutcome specialized_compact_turn_source_" << sourceIndex << "(\n"
            << "    const Game& game,\n"
            << "    PersistentLevelState& levelState,\n"
            << "    Scratch& scratch,\n"
            << "    SpecializedCompactTurnContext context,\n"
            << "    ps_input input,\n"
            << "    RuntimeStepOptions options\n"
            << ") {\n";
        if (!compactTurnSupport.supported) {
            out << "    (void)game;\n"
                << "    (void)levelState;\n"
                << "    (void)scratch;\n"
                << "    (void)context;\n"
                << "    (void)options;\n"
                << "    (void)input;\n"
                << "    return {false, {}};\n"
                << "}\n\n";
        } else if (compactTurnSupport.interpreterBridge) {
            out << "    return compactStateInterpretedTurnBridge(game, levelState, scratch, context, input, options);\n"
                << "}\n\n";
        } else {
            out << "    (void)game;\n"
                << "    (void)levelState;\n"
                << "    (void)scratch;\n"
                << "    (void)context;\n"
                << "    (void)options;\n"
                << "    (void)input;\n"
                << "    return {false, {}};\n"
                << "}\n\n";
        }
        out
            << "const SpecializedCompactTurnBackend specialized_compact_turn_backend_" << sourceIndex << " = {\n"
            << "    " << source.hash << "ULL,\n"
            << "    " << cppStringLiteral(source.path.string()) << ",\n"
            << "    specialized_compact_turn_source_" << sourceIndex << ",\n"
            << "    {" << (compactTurnSupport.supported ? "true" : "false")
            << ", " << cppStringLiteral(compactTurnSupport.fallbackReason) << "},\n"
            << "    " << (compactTurnSupport.supported && !compactTurnSupport.interpreterBridge ? "true" : "false") << ",\n"
            << "};\n\n";
    }

    out << "} // namespace\n\n";
    if (emitGlobalFinder) {
        out << "extern \"C\" const puzzlescript::SpecializedRulegroupsBackend* "
            << "ps_specialized_rulegroups_find_backend(uint64_t sourceHash) {\n"
            << "    switch (sourceHash) {\n";
        for (size_t sourceIndex = 0; sourceIndex < sources.size(); ++sourceIndex) {
            out << "        case " << sources[sourceIndex].hash << "ULL: return &backend_" << sourceIndex << ";\n";
        }
        out << "        default: return nullptr;\n"
            << "    }\n"
            << "}\n\n"
            << "extern \"C\" const puzzlescript::SpecializedRulegroupsBackend* "
            << "ps_compiled_rules_find_backend(uint64_t sourceHash) {\n"
            << "    return ps_specialized_rulegroups_find_backend(sourceHash);\n"
            << "}\n\n";
        out << "extern \"C\" const puzzlescript::SpecializedFullTurnBackend* "
            << "ps_specialized_full_turn_find_backend(uint64_t sourceHash) {\n"
            << "    switch (sourceHash) {\n";
        for (size_t sourceIndex = 0; sourceIndex < sources.size(); ++sourceIndex) {
            out << "        case " << sources[sourceIndex].hash << "ULL: return &tick_backend_" << sourceIndex << ";\n";
        }
        out << "        default: return nullptr;\n"
            << "    }\n"
            << "}\n\n"
            << "extern \"C\" const puzzlescript::SpecializedFullTurnBackend* "
            << "ps_compiled_tick_find_backend(uint64_t sourceHash) {\n"
            << "    return ps_specialized_full_turn_find_backend(sourceHash);\n"
            << "}\n\n";
        out << "extern \"C\" const puzzlescript::SpecializedCompactTurnBackend* "
            << "ps_specialized_compact_turn_find_backend(uint64_t sourceHash) {\n"
            << "    switch (sourceHash) {\n";
        for (size_t sourceIndex = 0; sourceIndex < sources.size(); ++sourceIndex) {
            out << "        case " << sources[sourceIndex].hash << "ULL: return &specialized_compact_turn_backend_" << sourceIndex << ";\n";
        }
        out << "        default: return nullptr;\n"
            << "    }\n"
            << "}\n\n"
            << "extern \"C\" const puzzlescript::SpecializedCompactTurnBackend* "
            << "ps_compiled_compact_tick_find_backend(uint64_t sourceHash) {\n"
            << "    return ps_specialized_compact_turn_find_backend(sourceHash);\n"
            << "}\n\n";
    } else {
        if (sources.size() != 1) {
            throw std::runtime_error("sharded compiled-rules codegen requires one source per generated C++ file");
        }
        const std::string safeBackendAccessor = safeCppIdentifier(
            backendAccessorSymbol.empty()
                ? std::string(safeSymbol + "_backend")
                : std::string(backendAccessorSymbol)
        );
        out << "extern \"C\" const puzzlescript::SpecializedRulegroupsBackend* "
            << safeBackendAccessor << "() {\n"
            << "    return &backend_0;\n"
            << "}\n\n";
        const std::string safeTickBackendAccessor = safeCppIdentifier(
            tickBackendAccessorSymbol.empty()
                ? std::string(safeSymbol + "_specialized_full_turn_backend")
                : std::string(tickBackendAccessorSymbol)
        );
        out << "extern \"C\" const puzzlescript::SpecializedFullTurnBackend* "
            << safeTickBackendAccessor << "() {\n"
            << "    return &tick_backend_0;\n"
            << "}\n\n";
        const std::string safeCompactTurnBackendAccessor = safeCppIdentifier(
            compactTurnBackendAccessorSymbol.empty()
                ? std::string(safeSymbol + "_specialized_compact_turn_backend")
                : std::string(compactTurnBackendAccessorSymbol)
        );
        out << "extern \"C\" const puzzlescript::SpecializedCompactTurnBackend* "
            << safeCompactTurnBackendAccessor << "() {\n"
            << "    return &specialized_compact_turn_backend_0;\n"
            << "}\n\n";
    }
    out << "extern \"C\" const uint32_t " << safeSymbol << "_compiled_rule_count = " << totalCompiledRules << "U;\n"
        << "extern \"C\" const uint32_t " << safeSymbol << "_compiled_group_count = " << totalCompiledGroups << "U;\n";
    return out.str();
}

std::string generateCompiledRulesRegistryCpp(
    const std::vector<CodegenSource>& sources,
    std::string_view symbol,
    const std::vector<std::string>& backendAccessorSymbols,
    const std::vector<std::string>& tickBackendAccessorSymbols,
    const std::vector<std::string>& compactTurnBackendAccessorSymbols,
    const CompiledRulesCoverage& coverage
) {
    std::ostringstream out;
    const std::string safeSymbol = safeCppIdentifier(symbol);
    out << "// Generated by puzzlescript_cpp specialize-rulegroups. Do not edit by hand.\n"
        << "#include <cstdint>\n"
        << "#include \"runtime/compiled_rules.hpp\"\n\n";
    for (const auto& backendSymbol : backendAccessorSymbols) {
        out << "extern \"C\" const puzzlescript::SpecializedRulegroupsBackend* "
            << safeCppIdentifier(backendSymbol) << "();\n";
    }
    for (const auto& backendSymbol : tickBackendAccessorSymbols) {
        out << "extern \"C\" const puzzlescript::SpecializedFullTurnBackend* "
            << safeCppIdentifier(backendSymbol) << "();\n";
    }
    for (const auto& backendSymbol : compactTurnBackendAccessorSymbols) {
        out << "extern \"C\" const puzzlescript::SpecializedCompactTurnBackend* "
            << safeCppIdentifier(backendSymbol) << "();\n";
    }
    out << "\nextern \"C\" const puzzlescript::SpecializedRulegroupsBackend* "
        << "ps_specialized_rulegroups_find_backend(uint64_t sourceHash) {\n"
        << "    switch (sourceHash) {\n";
    for (size_t sourceIndex = 0; sourceIndex < sources.size(); ++sourceIndex) {
        out << "        case " << sources[sourceIndex].hash << "ULL: return "
            << safeCppIdentifier(backendAccessorSymbols[sourceIndex]) << "();\n";
    }
    out << "        default: return nullptr;\n"
        << "    }\n"
        << "}\n\n"
        << "extern \"C\" const puzzlescript::SpecializedRulegroupsBackend* "
        << "ps_compiled_rules_find_backend(uint64_t sourceHash) {\n"
        << "    return ps_specialized_rulegroups_find_backend(sourceHash);\n"
        << "}\n\n";
    out << "extern \"C\" const puzzlescript::SpecializedFullTurnBackend* "
        << "ps_specialized_full_turn_find_backend(uint64_t sourceHash) {\n"
        << "    switch (sourceHash) {\n";
    for (size_t sourceIndex = 0; sourceIndex < sources.size(); ++sourceIndex) {
        out << "        case " << sources[sourceIndex].hash << "ULL: return "
            << safeCppIdentifier(tickBackendAccessorSymbols[sourceIndex]) << "();\n";
    }
    out << "        default: return nullptr;\n"
        << "    }\n"
        << "}\n\n"
        << "extern \"C\" const puzzlescript::SpecializedFullTurnBackend* "
        << "ps_compiled_tick_find_backend(uint64_t sourceHash) {\n"
        << "    return ps_specialized_full_turn_find_backend(sourceHash);\n"
        << "}\n\n";
    out << "extern \"C\" const puzzlescript::SpecializedCompactTurnBackend* "
        << "ps_specialized_compact_turn_find_backend(uint64_t sourceHash) {\n"
        << "    switch (sourceHash) {\n";
    for (size_t sourceIndex = 0; sourceIndex < sources.size(); ++sourceIndex) {
        out << "        case " << sources[sourceIndex].hash << "ULL: return "
            << safeCppIdentifier(compactTurnBackendAccessorSymbols[sourceIndex]) << "();\n";
    }
    out << "        default: return nullptr;\n"
        << "    }\n"
        << "}\n\n"
        << "extern \"C\" const puzzlescript::SpecializedCompactTurnBackend* "
        << "ps_compiled_compact_tick_find_backend(uint64_t sourceHash) {\n"
        << "    return ps_specialized_compact_turn_find_backend(sourceHash);\n"
        << "}\n\n"
        << "extern \"C\" const uint32_t " << safeSymbol << "_compiled_rule_count = " << coverage.compiledRules << "U;\n"
        << "extern \"C\" const uint32_t " << safeSymbol << "_compiled_group_count = " << coverage.compiledGroups << "U;\n";
    return out.str();
}

uint32_t writeCompiledRulesCppDirectory(
    const std::vector<CodegenSource>& sources,
    const std::filesystem::path& emitCppDir,
    const std::filesystem::path& emitSourcesList,
    std::string_view symbol,
    const CompiledRulesOptions& options,
    const CompiledRulesCoverage& coverage,
    bool compactTurnOnly = false
) {
    const std::string safeSymbol = safeCppIdentifier(symbol);
    std::vector<std::filesystem::path> generatedPaths;
    std::vector<std::string> backendAccessorSymbols;
    std::vector<std::string> tickBackendAccessorSymbols;
    std::vector<std::string> compactTurnBackendAccessorSymbols;
    uint32_t wroteCount = 0;
    generatedPaths.reserve(sources.size() + 1);
    backendAccessorSymbols.reserve(sources.size());
    tickBackendAccessorSymbols.reserve(sources.size());
    compactTurnBackendAccessorSymbols.reserve(sources.size());

    for (size_t sourceIndex = 0; sourceIndex < sources.size(); ++sourceIndex) {
        std::ostringstream hashHex;
        hashHex << std::hex << sources[sourceIndex].hash;
        const std::string hashSuffix = hashHex.str();
        const std::filesystem::path cppPath = emitCppDir / ("source_" + hashSuffix + ".cpp");
        const std::string sourceSymbol = safeSymbol + "_source_" + hashSuffix;
        const std::string backendSymbol = safeSymbol + "_backend_" + hashSuffix;
        const std::string tickBackendSymbol = safeSymbol + "_tick_backend_" + hashSuffix;
        const std::string compactTurnBackendSymbol = safeSymbol + "_specialized_compact_turn_backend_" + hashSuffix;
        backendAccessorSymbols.push_back(backendSymbol);
        tickBackendAccessorSymbols.push_back(tickBackendSymbol);
        compactTurnBackendAccessorSymbols.push_back(compactTurnBackendSymbol);
        const std::string generated = generateCompiledRulesCpp(
            std::vector<CodegenSource>{sources[sourceIndex]},
            sourceSymbol,
            options,
            false,
            backendSymbol,
            tickBackendSymbol,
            compactTurnBackendSymbol,
            compactTurnOnly
        );
        if (writeFileIfChanged(cppPath, generated)) {
            ++wroteCount;
        }
        generatedPaths.push_back(std::filesystem::absolute(cppPath));
    }

    const std::filesystem::path registryPath = emitCppDir / "registry.cpp";
    if (writeFileIfChanged(
        registryPath,
        generateCompiledRulesRegistryCpp(
            sources,
            symbol,
            backendAccessorSymbols,
            tickBackendAccessorSymbols,
            compactTurnBackendAccessorSymbols,
            compactTurnOnly ? CompiledRulesCoverage{} : coverage
        )
    )) {
        ++wroteCount;
    }
    generatedPaths.push_back(std::filesystem::absolute(registryPath));

    std::ostringstream list;
    for (const auto& path : generatedPaths) {
        list << path.string() << "\n";
    }
    if (writeFileIfChanged(emitSourcesList, list.str())) {
        ++wroteCount;
    }
    return wroteCount;
}

std::vector<CodegenSource> selectCompiledRuleSourcesForEmission(
    const std::vector<CodegenSource>& sources,
    const CompiledRulesOptions& options,
    std::optional<uint64_t> maxCompiledRulesPerSource,
    std::optional<uint64_t> maxGeneratedLinesPerSource,
    uint64_t& skippedSources,
    uint64_t& skippedCompiledRules,
    uint64_t& skippedGeneratedLineSources,
    uint64_t& skippedGeneratedLines
) {
    skippedSources = 0;
    skippedCompiledRules = 0;
    skippedGeneratedLineSources = 0;
    skippedGeneratedLines = 0;
    if (!maxCompiledRulesPerSource.has_value() && !maxGeneratedLinesPerSource.has_value()) {
        return sources;
    }

    std::vector<CodegenSource> selected;
    selected.reserve(sources.size());
    for (const auto& source : sources) {
        const CompiledRulesCoverage sourceCoverage = measureCompiledRulesCoverage(
            std::vector<CodegenSource>{source},
            options
        );
        if (sourceCoverage.compiledRules > *maxCompiledRulesPerSource) {
            ++skippedSources;
            skippedCompiledRules += sourceCoverage.compiledRules;
            continue;
        }
        if (maxGeneratedLinesPerSource.has_value()) {
            const std::string generated = generateCompiledRulesCpp(
                std::vector<CodegenSource>{source},
                "compiled_rules_line_budget_probe",
                options,
                false,
                "compiled_rules_line_budget_probe_backend",
                "compiled_rules_line_budget_probe_tick_backend",
                "compiled_rules_line_budget_probe_specialized_compact_turn_backend"
            );
            const uint64_t generatedLines = static_cast<uint64_t>(
                std::count(generated.begin(), generated.end(), '\n') + 1
            );
            if (generatedLines > *maxGeneratedLinesPerSource) {
                ++skippedSources;
                ++skippedGeneratedLineSources;
                skippedGeneratedLines += generatedLines;
                continue;
            }
        }
        selected.push_back(source);
    }
    return selected;
}

int compileRulesCommand(const std::string& sourcePath, int argc, char** argv) {
    std::optional<std::filesystem::path> emitCpp;
    std::optional<std::filesystem::path> emitCppDir;
    std::optional<std::filesystem::path> emitSourcesList;
    std::optional<std::filesystem::path> coverageJson;
    std::optional<uint64_t> maxCompiledRulesPerSource;
    std::optional<uint64_t> maxGeneratedLinesPerSource;
    std::string symbol = "puzzlescript_compiled_rules";
    CompiledRulesOptions options;
    bool statsOnly = false;
    bool compactTurnOnly = false;
    for (int index = 0; index < argc; ++index) {
        const std::string arg = argv[index];
        if (arg == "--emit-cpp") {
            if (++index >= argc) {
                throw std::runtime_error("--emit-cpp requires a path");
            }
            emitCpp = std::filesystem::path(argv[index]);
        } else if (arg == "--emit-cpp-dir") {
            if (++index >= argc) {
                throw std::runtime_error("--emit-cpp-dir requires a path");
            }
            emitCppDir = std::filesystem::path(argv[index]);
        } else if (arg == "--emit-sources-list") {
            if (++index >= argc) {
                throw std::runtime_error("--emit-sources-list requires a path");
            }
            emitSourcesList = std::filesystem::path(argv[index]);
        } else if (arg == "--coverage-json") {
            if (++index >= argc) {
                throw std::runtime_error("--coverage-json requires a path");
            }
            coverageJson = std::filesystem::path(argv[index]);
        } else if (arg == "--symbol") {
            if (++index >= argc) {
                throw std::runtime_error("--symbol requires a name");
            }
            symbol = argv[index];
        } else if (arg == "--stats-only") {
            statsOnly = true;
        } else if (arg == "--compact-turn-only" || arg == "--compact-tick-only") {
            compactTurnOnly = true;
        } else if (arg == "--max-rows") {
            if (++index >= argc) {
                throw std::runtime_error("--max-rows requires a positive integer");
            }
            const int parsed = std::stoi(argv[index]);
            if (parsed < 1) {
                throw std::runtime_error("--max-rows requires a positive integer");
            }
            options.maxRows = static_cast<size_t>(parsed);
        } else if (arg == "--max-compiled-rules-per-source") {
            if (++index >= argc) {
                throw std::runtime_error("--max-compiled-rules-per-source requires a non-negative integer");
            }
            const uint64_t parsed = static_cast<uint64_t>(std::stoull(argv[index]));
            if (parsed == 0) {
                maxCompiledRulesPerSource.reset();
            } else {
                maxCompiledRulesPerSource = parsed;
            }
        } else if (arg == "--max-generated-lines-per-source") {
            if (++index >= argc) {
                throw std::runtime_error("--max-generated-lines-per-source requires a non-negative integer");
            }
            const uint64_t parsed = static_cast<uint64_t>(std::stoull(argv[index]));
            if (parsed == 0) {
                maxGeneratedLinesPerSource.reset();
            } else {
                maxGeneratedLinesPerSource = parsed;
            }
        } else {
            throw std::runtime_error("Unsupported compile-rules argument: " + arg + "\nTry: puzzlescript_cpp help compile-rules");
        }
    }
    if (!statsOnly) {
        if (emitCpp.has_value() == emitCppDir.has_value()) {
            throw std::runtime_error("compile-rules requires exactly one of --emit-cpp out.cpp or --emit-cpp-dir out-dir");
        }
        if (emitCppDir.has_value() && !emitSourcesList.has_value()) {
            throw std::runtime_error("compile-rules --emit-cpp-dir requires --emit-sources-list out.txt");
        }
    }

    const std::vector<CodegenSource> inputSources = collectCompiledRuleSources(std::filesystem::path(sourcePath));
    uint64_t skippedSources = 0;
    uint64_t skippedCompiledRules = 0;
    uint64_t skippedGeneratedLineSources = 0;
    uint64_t skippedGeneratedLines = 0;
    std::vector<CodegenSource> sources = selectCompiledRuleSourcesForEmission(
        inputSources,
        options,
        maxCompiledRulesPerSource,
        maxGeneratedLinesPerSource,
        skippedSources,
        skippedCompiledRules,
        skippedGeneratedLineSources,
        skippedGeneratedLines
    );
    const CompiledRulesCoverage coverage = measureCompiledRulesCoverage(sources, options);
    if (coverageJson.has_value()) {
        writeFileIfChanged(
            *coverageJson,
            generateCompiledRulesCoverageJson(sources, options, coverage)
        );
    }
    if (statsOnly) {
        std::cerr << "compiled-rules: sources=" << sources.size()
                  << " max_rows=" << options.maxRows
                  << " output=<stats-only>"
                  << " wrote=0"
                  << "\n";
        if (maxCompiledRulesPerSource.has_value() && skippedCompiledRules > 0) {
            std::cerr << "compiled-rules-skips:"
                      << " max_compiled_rules_per_source=" << *maxCompiledRulesPerSource
                      << " skipped_sources=" << (skippedSources - skippedGeneratedLineSources)
                      << " skipped_compiled_rules=" << skippedCompiledRules
                      << "\n";
        }
        if (maxGeneratedLinesPerSource.has_value() && skippedGeneratedLineSources > 0) {
            std::cerr << "compiled-rules-line-skips:"
                      << " max_generated_lines_per_source=" << *maxGeneratedLinesPerSource
                      << " skipped_sources=" << skippedGeneratedLineSources
                      << " skipped_generated_lines=" << skippedGeneratedLines
                      << "\n";
        }
        printCompiledRulesCoverage(coverage);
        return 0;
    }

    uint32_t wroteCount = 0;
    std::string outputPath;
    if (emitCpp.has_value()) {
        const std::string generated = generateCompiledRulesCpp(sources, symbol, options, true, {}, {}, {}, compactTurnOnly);
        wroteCount = writeFileIfChanged(*emitCpp, generated) ? 1U : 0U;
        outputPath = emitCpp->string();
    } else {
        wroteCount = writeCompiledRulesCppDirectory(
            sources,
            *emitCppDir,
            *emitSourcesList,
            symbol,
            options,
            coverage,
            compactTurnOnly
        );
        outputPath = emitCppDir->string();
    }
    std::cerr << "compiled-rules: sources=" << sources.size()
              << " input_sources=" << inputSources.size()
              << " max_rows=" << options.maxRows
              << " groups=" << coverage.compiledGroups
              << " rules=" << coverage.compiledRules
              << (compactTurnOnly ? " compact_turn_only=true" : "")
              << " output=" << outputPath
              << " wrote=" << wroteCount
              << "\n";
    if (maxCompiledRulesPerSource.has_value() && skippedCompiledRules > 0) {
        std::cerr << "compiled-rules-skips:"
                  << " max_compiled_rules_per_source=" << *maxCompiledRulesPerSource
                  << " skipped_sources=" << (skippedSources - skippedGeneratedLineSources)
                  << " skipped_compiled_rules=" << skippedCompiledRules
                  << "\n";
    }
    if (maxGeneratedLinesPerSource.has_value() && skippedGeneratedLineSources > 0) {
        std::cerr << "compiled-rules-line-skips:"
                  << " max_generated_lines_per_source=" << *maxGeneratedLinesPerSource
                  << " skipped_sources=" << skippedGeneratedLineSources
                  << " skipped_generated_lines=" << skippedGeneratedLines
                  << "\n";
    }
    printCompiledRulesCoverage(coverage);
    return 0;
}

int compileSourceCommand(const std::string& sourcePath, int argc, char** argv) {
    bool emitParserState = false;
    bool emitDiagnostics = false;
    bool emitRuntimeIr = false;
    for (int index = 0; index < argc; ++index) {
        const std::string arg = argv[index];
        if (arg == "--emit-parser-state") {
            emitParserState = true;
        } else if (arg == "--emit-runtime-ir" || arg == "--emit-ir-json") {
            emitRuntimeIr = true;
        } else if (arg == "--diagnostics" || arg == "--emit-diagnostics") {
            emitDiagnostics = true;
        } else {
            throw std::runtime_error("Unsupported compile argument: " + arg + "\nTry: puzzlescript_cpp help compile");
        }
    }

    if (!emitParserState && !emitDiagnostics && !emitRuntimeIr) {
        std::cerr << "compile requires --diagnostics, --emit-parser-state, or --emit-ir-json.\n"
                  << "Try: puzzlescript_cpp compile " << sourcePath << " --diagnostics\n";
        return 1;
    }

    const std::string source = readFile(sourcePath) + "\n";
    std::unique_ptr<ps_compiler_result, decltype(&ps_compiler_result_free)> result(
        (emitDiagnostics
            ? ps_compiler_compile_source_diagnostics(source.data(), source.size())
            : ps_compiler_parse_source(source.data(), source.size())),
        ps_compiler_result_free
    );
    if (!result) {
        std::cerr << "Failed to parse source.\n";
        return 1;
    }

    if (emitDiagnostics) {
        for (size_t index = 0; index < ps_compiler_result_diagnostic_count(result.get()); ++index) {
            const ps_diagnostic* diagnostic = ps_compiler_result_diagnostic(result.get(), index);
            if (diagnostic == nullptr || diagnostic->message == nullptr) {
                continue;
            }
            std::cout << jsonStringLiteral(diagnostic->message) << "\n";
        }
    }

    if (emitParserState) {
        const size_t required = ps_compiler_result_parser_state_json(result.get(), nullptr, 0);
        std::string payload(required == 0 ? 0 : (required - 1), '\0');
        if (required > 0) {
            (void)ps_compiler_result_parser_state_json(result.get(), payload.data(), required);
        }
        std::cout << payload << "\n";
    }

    if (emitRuntimeIr) {
        puzzlescript::compiler::DiagnosticSink diagnostics;
        const auto parserState = puzzlescript::compiler::parseSource(source, diagnostics);
        puzzlescript::LoadedGame loadedGame;
        if (auto error = puzzlescript::compiler::lowerToRuntimeGame(parserState, loadedGame)) {
            std::cerr << error->message << "\n";
            return 1;
        }
        if (!loadedGame.information) {
            std::cerr << "Lowering produced no runtime game.\n";
            return 1;
        }
        puzzlescript::attachLinkedCompiledRules(*std::const_pointer_cast<puzzlescript::Game>(loadedGame.information), source);
        std::cout << serializeRuntimeGameDebugJson(loadedGame);
    }

    return 0;
}

int stepSourceCommand(const std::string& sourcePath, int argc, char** argv) {
    std::vector<std::string> exporterArgs;
    std::vector<std::string> inputTokens;
    bool nativeCompile = false;

    for (int index = 0; index < argc; ++index) {
        const std::string arg = argv[index];
        if (arg == "--native-compile") {
            nativeCompile = true;
            continue;
        }
        if ((arg == "--level" || arg == "--seed") && index + 1 < argc) {
            exporterArgs.push_back(arg);
            exporterArgs.push_back(argv[++index]);
        } else if (arg == "--settle-again") {
            exporterArgs.push_back(arg);
        } else {
            inputTokens.push_back(arg);
        }
    }
    ensureDefaultSourceLoad(exporterArgs);

    ps_game* game = nullptr;
    if (!nativeCompile) {
        if (!loadGameFromJsonText(runIrExporterAndCaptureJson(sourcePath, exporterArgs), &game)) {
            return 1;
        }
    } else {
        if (!loadGameFromSourceFile(sourcePath, &game)) {
            return 1;
        }
    }
    const std::optional<std::string> sessionSeed = nativeCompile ? findArgValue(exporterArgs, "--seed") : std::nullopt;
    const int result = stepCommandForGame(game, inputTokens, sessionSeed);
    ps_free_game(game);
    return result;
}

int diffTraceSourceCommand(const std::string& sourcePath, int argc, char** argv) {
    std::vector<std::string> irArgs;
    std::vector<std::string> traceArgs;
    bool nativeCompile = false;

    for (int index = 0; index < argc; ++index) {
        const std::string arg = argv[index];
        if (arg == "--native-compile") {
            nativeCompile = true;
            continue;
        }
        if ((arg == "--level" || arg == "--seed" || arg == "--inputs-json" || arg == "--inputs-file") && index + 1 < argc) {
            const std::string value = argv[++index];
            if (arg == "--level" || arg == "--seed") {
                irArgs.push_back(arg);
                irArgs.push_back(value);
            }
            traceArgs.push_back(arg);
            traceArgs.push_back(value);
        } else if (arg == "--settle-again") {
            irArgs.push_back(arg);
        } else {
            throw std::runtime_error("Unsupported diff-trace-source argument: " + arg);
        }
    }

    ps_game* game = nullptr;
    if (!nativeCompile) {
        if (!loadGameFromJsonText(runIrExporterAndCaptureJson(sourcePath, irArgs), &game)) {
            return 1;
        }
    } else {
        if (!loadGameFromSourceFile(sourcePath, &game)) {
            return 1;
        }
    }
    const std::optional<std::string> sessionSeed = nativeCompile ? findArgValue(traceArgs, "--seed") : std::nullopt;
    std::optional<int32_t> levelToLoad;
    if (nativeCompile) {
        if (const std::optional<std::string> level = findArgValue(traceArgs, "--level"); level.has_value()) {
            levelToLoad = static_cast<int32_t>(std::stoi(*level));
        }
    }
    const int result = diffTraceAgainstSnapshots(
        game,
        loadTraceSnapshotsFromJsonText(runTraceExporterAndCaptureJson(sourcePath, traceArgs)),
        std::cerr,
        true,
        sessionSeed,
        levelToLoad
    );
    ps_free_game(game);
    return result;
}

int testFixturesCommand(const std::string& manifestPath, int argc, char** argv) {
    const auto manifestDir = std::filesystem::path(manifestPath).parent_path();
    size_t preparedPassed = 0;
    size_t preparedFailed = 0;
    size_t traceChecked = 0;
    size_t tracePassed = 0;
    size_t traceFailed = 0;
    size_t traceLimit = 0;
    size_t traceProgress = 0;
    bool traceAll = false;
    bool traceAllowFailures = false;
    bool traceQuiet = false;
    bool profileTimers = false;

    for (int index = 0; index < argc; ++index) {
        const std::string arg = argv[index];
        if (arg == "--trace-all") {
            traceAll = true;
        } else if (arg == "--trace-limit" && index + 1 < argc) {
            traceLimit = static_cast<size_t>(std::stoull(argv[++index]));
        } else if (arg == "--trace-progress" && index + 1 < argc) {
            traceProgress = static_cast<size_t>(std::stoull(argv[++index]));
        } else if (arg == "--trace-allow-failures") {
            traceAllowFailures = true;
        } else if (arg == "--trace-quiet") {
            traceQuiet = true;
        } else if (arg == "--profile-timers") {
            profileTimers = true;
        } else {
            throw std::runtime_error("Unsupported JS parity data validation argument: " + arg);
        }
    }

    int64_t profileGameReuseUs = 0;
    int64_t profileGameLoadUs = 0;
    size_t profileGamesReused = 0;
    size_t profileGamesLoaded = 0;
    int64_t profileSessionCreateUs = 0;
    int64_t profileSerializeUs = 0;
    int64_t profileTraceDiffUs = 0;
    const auto preparedWallStart = std::chrono::steady_clock::now();

    try {
        PsGameCache cache;
        const auto fixtures = parseSimulationFixtureManifest(manifestPath, manifestDir);
        for (size_t index = 0; index < fixtures.size(); ++index) {
            try {
                const auto& fixture = fixtures[index];
                const auto& irPath = fixture.irFile;
                const auto& name = fixture.name;
                const auto& expectedSerialized = fixture.initialSerializedLevel;

                const bool gameCached = cache.has(irPath);
                const auto gameAcquireStart = std::chrono::steady_clock::now();
                ps_game* game = cache.acquire(irPath);
                const auto gameAcquireUs = std::chrono::duration_cast<std::chrono::microseconds>(
                                               std::chrono::steady_clock::now() - gameAcquireStart)
                                               .count();
                if (profileTimers) {
                    if (gameCached) {
                        ++profileGamesReused;
                        profileGameReuseUs += gameAcquireUs;
                    } else {
                        ++profileGamesLoaded;
                        profileGameLoadUs += gameAcquireUs;
                    }
                }

                if (game == nullptr) {
                    ++preparedFailed;
                    continue;
                }
                ps_full_state* session = nullptr;
                ps_error* error = nullptr;
                const auto sessionStart = std::chrono::steady_clock::now();
                if (!ps_full_state_create(game, &session, &error)) {
                    std::cerr << name << ": " << ps_error_message(error) << "\n";
                    ps_free_error(error);
                    ++preparedFailed;
                    if (profileTimers) {
                        profileSessionCreateUs += std::chrono::duration_cast<std::chrono::microseconds>(
                                                       std::chrono::steady_clock::now() - sessionStart)
                                                       .count();
                    }
                    continue;
                }
                if (profileTimers) {
                    profileSessionCreateUs += std::chrono::duration_cast<std::chrono::microseconds>(
                                                   std::chrono::steady_clock::now() - sessionStart)
                                                   .count();
                }

                const auto serializeStart = std::chrono::steady_clock::now();
                char* serialized = ps_full_state_serialize_test_string(session);
                const std::string actual = serialized ? serialized : "";
                if (profileTimers) {
                    profileSerializeUs += std::chrono::duration_cast<std::chrono::microseconds>(
                                               std::chrono::steady_clock::now() - serializeStart)
                                               .count();
                }
                if (actual == expectedSerialized) {
                    ++preparedPassed;
                } else {
                    ++preparedFailed;
                    if (!traceQuiet) {
                        std::cerr << name << ": prepared session mismatch\n";
                    }
                }

                ps_string_free(serialized);
                ps_full_state_destroy(session);

                const bool shouldCheckTrace = fixture.traceFile.has_value()
                    && (traceAll || (traceLimit > 0 && traceChecked < traceLimit));
                if (shouldCheckTrace) {
                    ++traceChecked;
                    if (traceProgress > 0 && (traceChecked % traceProgress) == 0) {
                        std::cerr << "trace_progress checked=" << traceChecked
                                  << " passed=" << tracePassed
                                  << " failed=" << traceFailed
                                  << " current_case=" << name
                                  << "\n";
                    }
                    std::ostringstream traceErrors;
                    const auto traceDiffStart = std::chrono::steady_clock::now();
                    const int traceResult = diffTraceAgainstSnapshots(game, loadTraceSnapshots(*fixture.traceFile), traceErrors, false);
                    if (profileTimers) {
                        profileTraceDiffUs += std::chrono::duration_cast<std::chrono::microseconds>(
                                                  std::chrono::steady_clock::now() - traceDiffStart)
                                                  .count();
                    }
                    if (traceResult == 0) {
                        ++tracePassed;
                    } else {
                        ++traceFailed;
                        if (!traceQuiet) {
                            std::cerr << name << ": trace replay mismatch\n" << traceErrors.str();
                        }
                    }
                }

            } catch (const std::exception& error) {
                ++preparedFailed;
                if (!traceQuiet) {
                    std::cerr << "case #" << index << ": " << error.what() << "\n";
                }
            }
        }
    } catch (const std::exception& error) {
        std::cerr << error.what() << "\n";
        return 1;
    }

    std::cout << "test_cases_checked=" << (preparedPassed + preparedFailed)
              << " prepared_session_checks_passed=" << preparedPassed
              << " prepared_session_checks_failed=" << preparedFailed
              << " trace_replay_checked=" << traceChecked
              << " trace_replay_passed=" << tracePassed
              << " trace_replay_failed=" << traceFailed
              << "\n";

    if (profileTimers) {
        const auto preparedWallUs = std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::steady_clock::now() - preparedWallStart)
                                        .count();
        const auto usToMs = [](int64_t microseconds) -> int64_t {
            return (microseconds + 500) / 1000;
        };
        const size_t fixtureCount = preparedPassed + preparedFailed;
        std::cerr << "prepared_profile test_cases=" << fixtureCount << " wall_ms=" << usToMs(preparedWallUs) << " games_reused=" << profileGamesReused
                  << " games_loaded=" << profileGamesLoaded << " game_reuse_ms=" << usToMs(profileGameReuseUs)
                  << " game_load_ms=" << usToMs(profileGameLoadUs)
                  << " session_create_ms=" << usToMs(profileSessionCreateUs) << " serialize_test_string_ms=" << usToMs(profileSerializeUs)
                  << " optional_trace_diff_ms=" << usToMs(profileTraceDiffUs) << "\n";
    }

    const bool preparedOk = preparedFailed == 0;
    const bool traceOk = traceAllowFailures || traceFailed == 0;
    return (preparedOk && traceOk) ? 0 : 1;
}

void printMainHelp() {
    std::cout
        << "puzzlescript_cpp: C++ PuzzleScript compiler, runtime, test runner, and SDL player\n\n"
        << "Common commands:\n"
        << "  puzzlescript_cpp play game.txt\n"
        << "      Build/load a PuzzleScript source file and launch the SDL player.\n"
        << "  puzzlescript_cpp run game.txt --headless\n"
        << "      Load a PuzzleScript source file and print the current board serialization.\n"
        << "  puzzlescript_cpp run game.txt --headless --inputs-file inputs.json\n"
        << "      Replay inputs through the headless runtime and compare against the JS oracle trace.\n"
        << "  puzzlescript_cpp compile game.txt --diagnostics\n"
        << "      Run the C++ compiler parser diagnostics.\n"
        << "  puzzlescript_cpp compile game.txt --emit-parser-state\n"
        << "      Emit the canonical parser-state JSON used by parity tests.\n"
        << "  puzzlescript_cpp compile game.txt --emit-ir-json\n"
        << "      Emit the native lowered runtime game JSON used for JS-vs-C++ compiler diffs.\n"
        << "  puzzlescript_cpp specialize-rulegroups game.txt --emit-cpp build/compiled-rules/game.cpp\n"
        << "      Emit C++ specialized rulegroup kernels for build-time solver/generator specialization.\n"
        << "  puzzlescript_cpp test js-parity <generated-js-parity-data.json>\n"
        << "      Check saved replay cases generated from the original JavaScript test suite.\n"
        << "  puzzlescript_cpp test simulation-corpus src/tests/resources/testdata.js\n"
        << "      Run the C++ compiler/runtime directly against the simulation corpus.\n"
        << "  puzzlescript_cpp test diagnostics-corpus src/tests/resources/errormessage_testdata.js\n"
        << "      Run the C++ compiler directly against the diagnostics corpus.\n"
        << "  puzzlescript_cpp bench game.txt --iterations 10000 --threads 4\n"
        << "      Benchmark clone/hash/full-state operations for a source game.\n\n"
        << "  puzzlescript_cpp profile-simulations generated-js-parity-data.json --repeat 3\n"
        << "      Run a C++-only replay workload for profiler/hot-function analysis.\n\n"
        << "Project map:\n"
        << "  compiler: native/src/compiler\n"
        << "  runtime:  native/src/runtime\n"
        << "  player:   native/src/player\n"
        << "  CLI:      native/src/cli\n"
        << "  JS oracle/reference harness: src/tests/js_oracle\n\n"
        << "Makefile shortcuts:\n"
        << "  make build           Build build/native/puzzlescript_cpp\n"
        << "  make run game.txt    Build and play a game\n"
        << "  make ctest           Run fast C++ smoke/unit tests\n"
        << "  make js_parity_tests Run C++ vs original-JS parity tests\n"
        << "  make simulation_tests\n"
        << "      Run JS simulation tests, then mirrored C++ simulation parity\n"
        << "  make compilation_tests\n"
        << "      Run JS compiler tests, then mirrored C++ diagnostics parity\n"
        << "  make tests           Run the full native correctness suite\n\n"
        << "Detailed help:\n"
        << "  puzzlescript_cpp help play\n"
        << "  puzzlescript_cpp help run\n"
        << "  puzzlescript_cpp help compile\n"
        << "  puzzlescript_cpp help specialize-rulegroups\n"
        << "  puzzlescript_cpp help test\n"
        << "  puzzlescript_cpp help profile\n"
        << "  puzzlescript_cpp help bench\n";
}

void printPlayHelp() {
    std::cout
        << "Usage: puzzlescript_cpp play game.txt\n\n"
        << "Compiles a PuzzleScript source file with the native C++ compiler and opens it\n"
        << "in the SDL player. Use play-ir only for explicit IR/dev debugging.\n\n"
        << "Example:\n"
        << "  puzzlescript_cpp play src/demo/sokoban_basic.txt\n";
}

void printRunHelp() {
    std::cout
        << "Usage: puzzlescript_cpp run game.txt [--headless] [--level N] [--seed seed] [--settle-again] [--native-compile]\n"
        << "       puzzlescript_cpp run game.txt --headless --inputs-file inputs.json\n\n"
        << "Runs a source game without opening a window. With no inputs it prints the current\n"
        << "board serialization; with --inputs-file/--inputs-json it replays inputs and\n"
        << "compares against the original JavaScript oracle trace.\n";
}

void printCompileHelp() {
    std::cout
        << "Usage: puzzlescript_cpp compile game.txt [--diagnostics] [--emit-parser-state] [--emit-ir-json]\n\n"
        << "Runs the C++ PuzzleScript compiler parser. Use --diagnostics for JS-compatible\n"
        << "diagnostic text, --emit-parser-state for parser JSON, and --emit-ir-json for\n"
        << "the lowered native runtime game JSON used to debug compiler parity before board\n"
        << "simulation.\n\n"
        << "Examples:\n"
        << "  puzzlescript_cpp compile game.txt --diagnostics\n"
        << "  puzzlescript_cpp compile game.txt --emit-parser-state\n"
        << "  puzzlescript_cpp compile game.txt --emit-ir-json\n";
}

void printCompileRulesHelp() {
    std::cout
        << "Usage: puzzlescript_cpp specialize-rulegroups game.txt --emit-cpp out.cpp [--symbol name] [--max-rows N]\n"
        << "       puzzlescript_cpp compile-rules game.txt --emit-cpp out.cpp [--symbol name] [--max-rows N]\n"
        << "       puzzlescript_cpp compile-rules path/to/corpus-dir --emit-cpp out.cpp [--symbol name] [--max-rows N]\n"
        << "       puzzlescript_cpp compile-rules path/to/corpus-dir --emit-cpp-dir out-dir --emit-sources-list out.txt [--symbol name] [--max-rows N] [--compact-turn-only] [--max-compiled-rules-per-source N] [--max-generated-lines-per-source N]\n"
        << "       puzzlescript_cpp compile-rules path/to/corpus --stats-only [--max-rows N] [--coverage-json out.json]\n\n"
        << "Emits C++ specialized rulegroup kernels for conservative deterministic rulegroups.\n"
        << "compile-rules remains as the compatibility command name used by existing build scripts.\n"
        << "The generated file, or sharded source directory, is meant to be linked into\n"
        << "solver/generator builds through the Makefile SPECIALIZE=true workflow. Use\n"
        << "--stats-only to print coverage and miss buckets without writing generated code. --coverage-json writes per-source coverage. --max-rows defaults to 1;\n"
        << "higher values enable experimental deterministic multi-row kernels. --compact-turn-only emits compact turn backends and registry stubs without generated rule kernels, useful for compact oracle coverage builds.\n"
        << "--compact-tick-only remains as a compatibility alias. --max-compiled-rules-per-source and --max-generated-lines-per-source skip oversized sharded sources so the runtime can fall back for those games.\n\n"
        << "Examples:\n"
        << "  puzzlescript_cpp specialize-rulegroups src/demo/sokoban_basic.txt --emit-cpp build/compiled-rules/sokoban.cpp --symbol sokoban\n"
        << "  puzzlescript_cpp compile-rules src/tests/solver_tests --emit-cpp-dir build/compiled-rules/solver-tests --emit-sources-list build/compiled-rules/solver-tests.txt\n"
        << "  puzzlescript_cpp compile-rules src/tests/resources/testdata.js --stats-only --max-rows 8 --coverage-json build/compiled-rules/coverage.json\n"
        << "  make generator src/demo/sokoban_basic.txt src/tests/generator_presets/sokoban_room_scatter.gen SPECIALIZE=true\n";
}

void printTestHelp() {
    std::cout
        << "Usage: puzzlescript_cpp test js-parity generated-js-parity-data.json [options]\n"
        << "       puzzlescript_cpp test simulation-corpus src/tests/resources/testdata.js [--progress-every N] [--profile-timers] [--repeat N] [--jobs N|auto] [--top-slow-cases N] [--case-index N] [--case-name text] [--compact-turn-oracle] [--require-compact-turn-oracle-checks] [--quiet]\n"
        << "       puzzlescript_cpp test diagnostics-corpus src/tests/resources/errormessage_testdata.js [--progress-every N]\n"
        << "       puzzlescript_cpp test diagnostics parser-corpus.bundle.ndjson\n\n"
        << "simulation-corpus and diagnostics-corpus read the original testdata.js and\n"
        << "errormessage_testdata.js directly as JSON-ish arrays and do not invoke Node or\n"
        << "the JavaScript PuzzleScript engine. js-parity uses saved replay data generated\n"
        << "from the original JavaScript implementation for deeper runtime trace checks.\n\n"
        << "simulation-corpus profiling reports source compile/load/replay/serialize timings\n"
        << "plus runtime counters for rule scans, pattern tests, replacements, mask rebuilds, and specialized dispatch.\n\n"
        << "With --compact-turn-oracle, simulation-corpus checks linked generated compact turn\n"
        << "entrypoints against the interpreter before replaying each ordinary input. --compact-tick-oracle remains as a compatibility alias.\n\n"
        << "For optimization work, --top-slow-cases lists the slowest games by phase, and\n"
        << "--case-index/--case-name reruns one slow case with counters and repeats.\n\n"
        << "Usually use the Makefile wrappers:\n"
        << "  make js_parity_tests\n"
        << "  make simulation_tests\n"
        << "  make compilation_tests\n"
        << "  make simulation_tests_js / make simulation_tests_cpp\n"
        << "  make compilation_tests_js / make compilation_tests_cpp\n"
        << "  make tests\n";
}

void printProfileHelp() {
    std::cout
        << "Usage: puzzlescript_cpp test simulation-corpus src/tests/resources/testdata.js --profile-timers [--repeat N] [--jobs N|auto] [--top-slow-cases N] [--case-index N] [--case-name text] [--quiet]\n"
        << "       puzzlescript_cpp profile-simulations generated-js-parity-data.json [--repeat N] [--profile-timers] [--quiet]\n\n"
        << "The direct simulation-corpus profiler is the current performance north star: it\n"
        << "parses testdata.js directly, compiles games natively, replays inputs, and splits\n"
        << "time into source compile, full-state creation, replay, serialization, and runtime\n"
        << "rule/specialized-dispatch counters. profile-simulations remains available for generated JS parity\n"
        << "replay data when debugging trace-level behavior.\n\n"
        << "Examples:\n"
        << "  puzzlescript_cpp test simulation-corpus src/tests/resources/testdata.js --jobs 1 --top-slow-cases 10\n"
        << "  puzzlescript_cpp test simulation-corpus src/tests/resources/testdata.js --case-index 155 --repeat 5 --profile-timers\n\n"
        << "Usually use:\n"
        << "  make profile_simulation_tests\n";
}

void printBenchHelp() {
    std::cout
        << "Usage: puzzlescript_cpp bench game.txt [--level N] [--seed seed] [--settle-again] [--native-compile] [--iterations N] [--threads N]\n\n"
        << "Benchmarks native runtime operations for a source game. Use --threads to measure\n"
        << "multi-full-state throughput for future solver workloads.\n\n"
        << "Example:\n"
        << "  puzzlescript_cpp bench src/demo/sokoban_basic.txt --iterations 10000 --threads 4\n";
}

void printHelpTopic(const std::string& topic) {
    if (topic == "play") {
        printPlayHelp();
    } else if (topic == "run") {
        printRunHelp();
    } else if (topic == "compile") {
        printCompileHelp();
    } else if (topic == "compile-rules" || topic == "specialize-rulegroups") {
        printCompileRulesHelp();
    } else if (topic == "test") {
        printTestHelp();
    } else if (topic == "profile" || topic == "profile-simulations") {
        printProfileHelp();
    } else if (topic == "bench") {
        printBenchHelp();
    } else {
        printMainHelp();
    }
}

} // namespace

int main(int argc, char** argv) {
    if (argc < 2 || std::string(argv[1]) == "--help" || std::string(argv[1]) == "-h") {
        printMainHelp();
        return 0;
    }

    const std::string command = argv[1];
    if (command == "help") {
        printHelpTopic(argc >= 3 ? argv[2] : "");
        return 0;
    }
    if (command == "test" && argc >= 3 && (std::string(argv[2]) == "--help" || std::string(argv[2]) == "-h")) {
        printTestHelp();
        return 0;
    }

    if (argc < 3) {
        std::cerr << "Missing path or subcommand.\nTry: puzzlescript_cpp --help\n";
        return 1;
    }

    const std::string path = argv[2];

    try {
        if (command == "run") {
            return runSourceCommand(path, argc - 3, argv + 3);
        }
        if (command == "step") {
            return stepSourceCommand(path, argc - 3, argv + 3);
        }
        if (command == "bench") {
            return benchSourceCommand(path, argc - 3, argv + 3);
        }
        if (command == "compile") {
            return compileSourceCommand(path, argc - 3, argv + 3);
        }
        if (command == "compile-rules" || command == "specialize-rulegroups") {
            return compileRulesCommand(path, argc - 3, argv + 3);
        }
        if (command == "play") {
#ifdef PS_HAVE_SDL2
            return playSourceCommand(path, argc - 3, argv + 3);
#else
            std::cerr << "SDL2 support is not enabled in this build.\n";
            return 1;
#endif
        }
        if (command == "run-ir") {
            return runCommand(path, argc - 3, argv + 3);
        }
        if (command == "step-ir") {
            return stepCommand(path, argc - 3, argv + 3);
        }
        if (command == "bench-ir") {
            return benchCommand(path, argc - 3, argv + 3);
        }
        if (command == "play-ir") {
#ifdef PS_HAVE_SDL2
            return puzzlescript_cpp_run_player_for_ir(path);
#else
            std::cerr << "SDL2 support is not enabled in this build.\n";
            return 1;
#endif
        }
        if (command == "diff-trace") {
            if (argc >= 4 && std::filesystem::path(path).extension() == ".json") {
                return diffTraceCommand(path, argv[3]);
            }
            return diffTraceSourceCommand(path, argc - 3, argv + 3);
        }
        if (command == "check-trace") {
            if (argc < 4) {
                std::cerr << "Usage: puzzlescript_cpp check-trace <ir.json> <trace.json>\n";
                return 1;
            }
            return checkTraceCommand(path, argv[3]);
        }
        if (command == "check-js-parity-data") {
            return checkTraceSweepCommand(path, argc - 3, argv + 3);
        }
        if (command == "profile-simulations") {
            return profileSimulationsCommand(path, argc - 3, argv + 3);
        }
        if (command == "trace-at") {
            if (argc < 5) {
                std::cerr << "Usage: puzzlescript_cpp trace-at <ir.json> <trace.json> <snapshot-index>\n";
                return 1;
            }
            return traceAtCommand(path, argv[3], static_cast<size_t>(std::stoull(argv[4])));
        }
        if (command == "trace-step-at") {
            if (argc < 5) {
                std::cerr << "Usage: puzzlescript_cpp trace-step-at <ir.json> <trace.json> <snapshot-index>\n";
                return 1;
            }
            return traceStepAtCommand(path, argv[3], static_cast<size_t>(std::stoull(argv[4])));
        }
        if (command == "test-js-parity-data") {
            return testFixturesCommand(path, argc - 3, argv + 3);
        }
        if (command == "simulation-testdata") {
            return simulationTestdataCommand(std::filesystem::path(path), argc - 3, argv + 3);
        }
        if (command == "compilation-testdata") {
            return compilationTestdataCommand(std::filesystem::path(path), argc - 3, argv + 3);
        }
        if (command == "diagnostics-parity") {
            return diagnosticsParityMain(std::filesystem::path(path));
        }
        if (command == "test") {
            if (path == "js-parity" && argc >= 4) {
                return checkTraceSweepCommand(argv[3], argc - 4, argv + 4);
            }
            if (path == "simulation-corpus" && argc >= 4) {
                return simulationTestdataCommand(std::filesystem::path(argv[3]), argc - 4, argv + 4);
            }
            if (path == "diagnostics-corpus" && argc >= 4) {
                return compilationTestdataCommand(std::filesystem::path(argv[3]), argc - 4, argv + 4);
            }
            if (path == "diagnostics" && argc >= 4) {
                return diagnosticsParityMain(std::filesystem::path(argv[3]));
            }
            printTestHelp();
            return 1;
        }
    } catch (const std::exception& error) {
        std::cerr << error.what() << "\n";
        return 1;
    }

    std::cerr << "Unknown command: " << command << "\nTry: puzzlescript_cpp --help\n";
    return 1;
}
