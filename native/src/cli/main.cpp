#include <array>
#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <memory>
#include <optional>
#include <sstream>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>

#include "cli/diagnostics_parity.hpp"
#include "compiler/parser.hpp"
#include "compiler/lower_to_runtime.hpp"
#include "runtime/json.hpp"
#include "puzzlescript/compiler.h"
#include "puzzlescript/puzzlescript.h"

#ifdef PS_HAVE_SDL2
int puzzlescript_cpp_run_player_for_ir(const std::string& irPath);
int puzzlescript_cpp_run_player_for_game(ps_game* game);
#endif

namespace {

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

std::vector<std::string> loadInputTokensFromJsonFile(const std::filesystem::path& path) {
    return loadInputTokensFromJsonText(readFile(path));
}

bool replayInputTokens(ps_session* session, const std::vector<std::string>& tokens, std::vector<std::string>* outSounds) {
    if (!session) {
        return false;
    }
    if (outSounds) {
        outSounds->clear();
    }
    for (const auto& token : tokens) {
        if (token == "undo") {
            (void)ps_session_undo(session);
            continue;
        }
        if (token == "restart") {
            (void)ps_session_restart(session);
            continue;
        }
        int32_t inputValue = 0;
        try {
            inputValue = std::stoi(token);
        } catch (...) {
            inputValue = 0;
        }

        ps_step_result stepResult{};
        if (inputValue == static_cast<int32_t>(PS_INPUT_TICK)) {
            stepResult = ps_session_tick(session);
        } else {
            if (inputValue < 0) inputValue = 0;
            if (inputValue > static_cast<int32_t>(PS_INPUT_TICK)) inputValue = static_cast<int32_t>(PS_INPUT_TICK);
            stepResult = ps_session_step(session, static_cast<ps_input>(inputValue));
        }

        if (outSounds && stepResult.audio_event_count > 0 && stepResult.audio_events) {
            for (size_t i = 0; i < stepResult.audio_event_count; ++i) {
                const ps_audio_event& event = stepResult.audio_events[i];
                if (event.kind && event.kind[0] != '\0') {
                    outSounds->push_back(event.kind);
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

int printSession(ps_session* session) {
    char* serialized = ps_session_serialize_test_string(session);
    char* snapshot = ps_session_export_snapshot(session);
    const uint64_t hash64 = ps_session_hash64(session);
    const ps_hash128 hash128 = ps_session_hash128(session);

    std::cout << serialized;
    std::cout << "hash64=" << hash64 << "\n";
    std::cout << "hash128=" << hash128.lo << ":" << hash128.hi << "\n";
    std::cout << snapshot << "\n";

    ps_string_free(snapshot);
    ps_string_free(serialized);
    return 0;
}

int runCommandForGame(ps_game* game) {
    ps_session* session = nullptr;
    ps_error* error = nullptr;
    if (!ps_session_create(game, &session, &error)) {
        std::cerr << ps_error_message(error) << "\n";
        ps_free_error(error);
        return 1;
    }

    const int result = printSession(session);
    ps_session_destroy(session);
    return result;
}

int benchCommandForGame(ps_game* game, uint32_t iterations, uint32_t threads) {
    ps_session* session = nullptr;
    ps_error* error = nullptr;
    if (!ps_session_create(game, &session, &error)) {
        std::cerr << ps_error_message(error) << "\n";
        ps_free_error(error);
        return 1;
    }

    ps_benchmark_result result{};
    if (!ps_benchmark_clone_hash(session, iterations, threads, &result, &error)) {
        std::cerr << ps_error_message(error) << "\n";
        ps_free_error(error);
        ps_session_destroy(session);
        return 1;
    }

    std::cout << "iterations=" << result.iterations
              << " threads=" << result.threads
              << " elapsed_seconds=" << result.elapsed_seconds
              << " iterations_per_second=" << result.iterations_per_second
              << " hash_accumulator=" << result.hash_accumulator
              << "\n";

    ps_session_destroy(session);
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
    return std::nullopt;
}

int stepCommandForGame(ps_game* game, const std::vector<std::string>& inputTokens) {
    ps_session* session = nullptr;
    ps_error* error = nullptr;
    if (!ps_session_create(game, &session, &error)) {
        std::cerr << ps_error_message(error) << "\n";
        ps_free_error(error);
        return 1;
    }

    for (size_t index = 0; index < inputTokens.size(); ++index) {
        const auto& token = inputTokens[index];
        ps_step_result result{};
        if (token == "undo") {
            if (!ps_session_undo(session)) {
                std::cerr << "step[" << index << "] undo failed\n";
                ps_session_destroy(session);
                return 1;
            }
        } else if (token == "restart") {
            if (!ps_session_restart(session)) {
                std::cerr << "step[" << index << "] restart failed\n";
                ps_session_destroy(session);
                return 1;
            }
        } else {
            const auto input = parseInputToken(token);
            if (!input.has_value()) {
                std::cerr << "Unsupported step token: " << token << "\n";
                ps_session_destroy(session);
                return 1;
            }
            result = (*input == PS_INPUT_TICK) ? ps_session_tick(session) : ps_session_step(session, *input);
        }
        std::cout << "step[" << index << "] token=" << token
                  << " changed=" << (result.changed ? 1 : 0)
                  << " transitioned=" << (result.transitioned ? 1 : 0)
                  << " won=" << (result.won ? 1 : 0)
                  << " audio_events=" << result.audio_event_count
                  << "\n";
    }

    const int result = printSession(session);
    ps_session_destroy(session);
    return result;
}

bool compareSnapshot(const TraceSnapshot& expected, ps_session* session, const ps_step_result* stepResult, size_t snapshotIndex, std::ostream& stream) {
    ps_session_status_info status{};
    ps_session_status(session, &status);
    char* serialized = ps_session_serialize_test_string(session);
    const std::string actualSerialized = serialized ? serialized : "";
    ps_string_free(serialized);
    char* snapshotJson = ps_session_export_snapshot(session);
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
    if (stepResult) {
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
    } else if (!expected.newSounds.empty()) {
        stream << "snapshot[" << snapshotIndex << "] expected audio events but no step result was provided\n";
        ok = false;
    }
    return ok;
}

int diffTraceAgainstSnapshots(ps_game* game, const std::vector<TraceSnapshot>& snapshots, std::ostream& errorStream, bool printSuccessSummary) {
    ps_session* session = nullptr;
    ps_error* error = nullptr;
    if (!ps_session_create(game, &session, &error)) {
        errorStream << ps_error_message(error) << "\n";
        ps_free_error(error);
        return 1;
    }
    std::unique_ptr<ps_session, decltype(&ps_session_destroy)> sessionHolder(session, ps_session_destroy);

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
            stepResult = ps_session_tick(session);
        } else if (snapshot.numericInput.has_value()) {
            stepResult = ps_session_step(session, static_cast<ps_input>(*snapshot.numericInput));
        } else if (snapshot.stringInput.has_value()) {
            if (*snapshot.stringInput == "tick") {
                stepResult = ps_session_tick(session);
            } else if (*snapshot.stringInput == "restart") {
                if (!ps_session_restart(session)) {
                    errorStream << "Restart failed at snapshot[" << index << "]\n";
                    return 1;
                }
            } else if (*snapshot.stringInput == "undo") {
                if (!ps_session_undo(session)) {
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
    ps_session* session = nullptr;
    ps_error* error = nullptr;
    if (!ps_session_create(game, &session, &error)) {
        errorStream << ps_error_message(error) << "\n";
        ps_free_error(error);
        return 1;
    }
    std::unique_ptr<ps_session, decltype(&ps_session_destroy)> sessionHolder(session, ps_session_destroy);

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
            stepResult = ps_session_tick(session);
        } else if (snapshot.numericInput.has_value()) {
            stepResult = ps_session_step(session, static_cast<ps_input>(*snapshot.numericInput));
        } else if (snapshot.stringInput.has_value()) {
            if (*snapshot.stringInput == "tick") {
                stepResult = ps_session_tick(session);
            } else if (*snapshot.stringInput == "restart") {
                if (!ps_session_restart(session)) {
                    errorStream << "Restart failed at snapshot[" << index << "]\n";
                    return 1;
                }
            } else if (*snapshot.stringInput == "undo") {
                if (!ps_session_undo(session)) {
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
    char* serialized = ps_session_serialize_test_string(session);
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

bool replayTraceInputsOnly(ps_session* session, const TraceFile& traceFile, std::ostream& errorStream, size_t& replayedSteps) {
    const auto& snapshots = traceFile.snapshots;
    if (snapshots.empty()) {
        errorStream << "Trace has no snapshots\n";
        return false;
    }

    for (size_t index = 1; index < snapshots.size(); ++index) {
        const auto& snapshot = snapshots[index];
        if (snapshot.phase == "again") {
            (void)ps_session_tick(session);
            ++replayedSteps;
        } else if (snapshot.numericInput.has_value()) {
            (void)ps_session_step(session, static_cast<ps_input>(*snapshot.numericInput));
            ++replayedSteps;
        } else if (snapshot.stringInput.has_value()) {
            if (*snapshot.stringInput == "tick") {
                (void)ps_session_tick(session);
                ++replayedSteps;
            } else if (*snapshot.stringInput == "restart") {
                if (!ps_session_restart(session)) {
                    errorStream << "Restart failed at snapshot[" << index << "]\n";
                    return false;
                }
                ++replayedSteps;
            } else if (*snapshot.stringInput == "undo") {
                if (!ps_session_undo(session)) {
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
    ps_session* session = nullptr;
    ps_error* error = nullptr;
    const auto sessionStart = std::chrono::steady_clock::now();
    if (!ps_session_create(game, &session, &error)) {
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
    char* serialized = ps_session_serialize_test_string(session);
    const std::string actual = serialized ? serialized : "";
    if (profileTimers) {
        profileSerializePreparedUs += std::chrono::duration_cast<std::chrono::microseconds>(
                                           std::chrono::steady_clock::now() - serializeStart)
                                           .count();
    }
    ps_string_free(serialized);
    ps_session_destroy(session);

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
                ps_session* session = nullptr;
                ps_error* error = nullptr;
                const auto sessionStart = std::chrono::steady_clock::now();
                if (!ps_session_create(game, &session, &error)) {
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
                std::unique_ptr<ps_session, decltype(&ps_session_destroy)> sessionHolder(session, ps_session_destroy);
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

    ps_session* session = nullptr;
    ps_error* error = nullptr;
    if (!ps_session_create(game, &session, &error)) {
        std::cerr << ps_error_message(error) << "\n";
        ps_free_error(error);
        ps_free_game(game);
        return 1;
    }

    for (size_t index = 1; index <= snapshotIndex; ++index) {
        const auto& snapshot = snapshots[index];
        if (snapshot.phase == "again") {
            (void)ps_session_tick(session);
        } else if (snapshot.numericInput.has_value()) {
            (void)ps_session_step(session, static_cast<ps_input>(*snapshot.numericInput));
        } else if (snapshot.stringInput.has_value()) {
            if (*snapshot.stringInput == "tick") {
                (void)ps_session_tick(session);
            } else if (*snapshot.stringInput == "restart") {
                if (!ps_session_restart(session)) {
                    std::cerr << "Restart failed at snapshot[" << index << "]\n";
                    ps_session_destroy(session);
                    ps_free_game(game);
                    return 1;
                }
            } else if (*snapshot.stringInput == "undo") {
                if (!ps_session_undo(session)) {
                    std::cerr << "Undo failed at snapshot[" << index << "]\n";
                    ps_session_destroy(session);
                    ps_free_game(game);
                    return 1;
                }
            } else {
                std::cerr << "Unsupported trace input token: " << *snapshot.stringInput << "\n";
                ps_session_destroy(session);
                ps_free_game(game);
                return 1;
            }
        } else {
            std::cerr << "Snapshot[" << index << "] has no replayable input token\n";
            ps_session_destroy(session);
            ps_free_game(game);
            return 1;
        }
    }

    const int result = printSession(session);
    ps_session_destroy(session);
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

    ps_session* session = nullptr;
    ps_error* error = nullptr;
    if (!ps_session_create(game, &session, &error)) {
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
                (void)ps_session_tick(session);
            } else if (snapshot.numericInput.has_value()) {
                (void)ps_session_step(session, static_cast<ps_input>(*snapshot.numericInput));
            } else if (snapshot.stringInput.has_value()) {
                if (*snapshot.stringInput == "tick") {
                    (void)ps_session_tick(session);
                } else if (*snapshot.stringInput == "restart") {
                    if (!ps_session_restart(session)) {
                        std::cerr << "Restart failed at snapshot[" << index << "]\n";
                        ps_session_destroy(session);
                        ps_free_game(game);
                        return 1;
                    }
                } else if (*snapshot.stringInput == "undo") {
                    if (!ps_session_undo(session)) {
                        std::cerr << "Undo failed at snapshot[" << index << "]\n";
                        ps_session_destroy(session);
                        ps_free_game(game);
                        return 1;
                    }
                } else {
                    std::cerr << "Unsupported trace input token: " << *snapshot.stringInput << "\n";
                    ps_session_destroy(session);
                    ps_free_game(game);
                    return 1;
                }
            } else {
                std::cerr << "Snapshot[" << index << "] has no replayable input token\n";
                ps_session_destroy(session);
                ps_free_game(game);
                return 1;
            }
        }
    }

    const auto& nextSnapshot = snapshots[snapshotIndex + 1];
    ps_step_result stepResult{};
    if (nextSnapshot.phase == "again") {
        stepResult = ps_session_tick(session);
    } else if (nextSnapshot.numericInput.has_value()) {
        stepResult = ps_session_step(session, static_cast<ps_input>(*nextSnapshot.numericInput));
    } else if (nextSnapshot.stringInput.has_value()) {
        if (*nextSnapshot.stringInput == "tick") {
            stepResult = ps_session_tick(session);
        } else if (*nextSnapshot.stringInput == "restart") {
            if (!ps_session_restart(session)) {
                std::cerr << "Restart failed at snapshot[" << (snapshotIndex + 1) << "]\n";
                ps_session_destroy(session);
                ps_free_game(game);
                return 1;
            }
        } else if (*nextSnapshot.stringInput == "undo") {
            if (!ps_session_undo(session)) {
                std::cerr << "Undo failed at snapshot[" << (snapshotIndex + 1) << "]\n";
                ps_session_destroy(session);
                ps_free_game(game);
                return 1;
            }
        } else {
            std::cerr << "Unsupported trace input token: " << *nextSnapshot.stringInput << "\n";
            ps_session_destroy(session);
            ps_free_game(game);
            return 1;
        }
    } else {
        std::cerr << "Snapshot[" << (snapshotIndex + 1) << "] has no replayable input token\n";
        ps_session_destroy(session);
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
    ps_session_destroy(session);
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

int runSourceCommand(const std::string& sourcePath, int argc, char** argv) {
    std::vector<std::string> exporterArgs;
    std::vector<std::string> traceArgs;
    bool hasInputTrace = false;
    bool finalOnly = false;
    bool emitJson = false;
    bool nativeCompile = false;
    std::optional<std::string> inputsJson;
    std::optional<std::string> inputsFile;
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
        result = diffTraceAgainstSnapshots(
            game,
            loadTraceSnapshotsFromJsonText(runTraceExporterAndCaptureJson(sourcePath, traceArgs)),
            std::cerr,
            true
        );
    } else {
        ps_session* session = nullptr;
        ps_error* error = nullptr;
        if (!ps_session_create(game, &session, &error)) {
            std::cerr << ps_error_message(error) << "\n";
            ps_free_error(error);
            ps_free_game(game);
            return 1;
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
            ps_session_destroy(session);
            ps_free_game(game);
            return 1;
        }

        char* serialized = ps_session_serialize_test_string(session);
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

        ps_session_destroy(session);
        result = 0;
    }
    ps_free_game(game);
    return result;
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
    const int result = benchCommandForGame(game, iterations, threads);
    ps_free_game(game);
    return result;
}

int playSourceCommand(const std::string& sourcePath, int argc, char** argv) {
    std::vector<std::string> exporterArgs;
    bool nativeCompile = false;
    for (int index = 0; index < argc; ++index) {
        const std::string arg = argv[index];
        if (arg == "--native-compile") {
            nativeCompile = true;
            continue;
        }
        exporterArgs.emplace_back(arg);
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
#ifdef PS_HAVE_SDL2
    const int result = puzzlescript_cpp_run_player_for_game(game);
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

std::string serializeRuntimeGameDebugJson(const puzzlescript::Game& game) {
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
    out << "    \"metadata_pairs\": "; appendJsonStringArray(out, game.metadataPairs); out << ",\n";
    // metadata_map / lines are optional; include to ease diffs.
    out << "    \"metadata_map\": {";
    {
        bool first = true;
        for (const auto& [k, v] : game.metadataMap) {
            if (!first) out << ",";
            first = false;
            out << jsonStringLiteral(k) << ":" << jsonStringLiteral(v);
        }
    }
    out << "},\n";
    out << "    \"metadata_lines\": {";
    {
        bool first = true;
        for (const auto& [k, v] : game.metadataLines) {
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
                out << ",\"commands\":[]";
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
                                // Required by parseReplacement even if unused.
                                out << ",\"random_entity_mask\":[]";
                                out << ",\"random_dir_mask\":[]";
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
    out << "    \"late_rules\": "; appendRules(game.lateRules); out << "\n";
    out << "  }";

    // prepared_session is optional, but emitting it makes native-vs-js diffs much easier.
    out << ",\n  \"prepared_session\": {";
    out << "\"current_level_index\":" << game.preparedSession.currentLevelIndex << ",";
    out << "\"current_level_target\":null,";
    out << "\"title_screen\":" << (game.preparedSession.titleScreen ? "true" : "false") << ",";
    out << "\"text_mode\":" << (game.preparedSession.textMode ? "true" : "false") << ",";
    out << "\"title_mode\":" << game.preparedSession.titleMode << ",";
    out << "\"title_selection\":" << game.preparedSession.titleSelection << ",";
    out << "\"title_selected\":" << (game.preparedSession.titleSelected ? "true" : "false") << ",";
    out << "\"message_selected\":" << (game.preparedSession.messageSelected ? "true" : "false") << ",";
    out << "\"winning\":" << (game.preparedSession.winning ? "true" : "false") << ",";
    out << "\"loaded_level_seed\":" << jsonStringLiteral(game.preparedSession.loadedLevelSeed) << ",";
    out << "\"random_state\":null,";
    out << "\"old_flickscreen_dat\":[],";
    out << "\"level\":";
    if (game.preparedSession.level.isMessage) {
        out << "{\"kind\":\"message\",\"message\":" << jsonStringLiteral(game.preparedSession.level.message) << "}";
    } else {
        out << "{\"kind\":\"level\",\"line_number\":" << game.preparedSession.level.lineNumber
            << ",\"width\":" << game.preparedSession.level.width
            << ",\"height\":" << game.preparedSession.level.height
            << ",\"layer_count\":" << game.preparedSession.level.layerCount
            << ",\"objects\":";
        appendJsonIntArray(out, game.preparedSession.level.objects);
        out << "}";
    }
    out << ",\"serialized_level\":" << jsonStringLiteral(game.preparedSession.serializedLevel);
    out << ",\"restart_target\":null";
    out << "}\n";
    out << "}\n";
    return out.str();
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
        ps_compiler_parse_source(source.data(), source.size()),
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
        std::shared_ptr<const puzzlescript::Game> game;
        if (auto error = puzzlescript::compiler::lowerToRuntimeGame(parserState, game)) {
            std::cerr << error->message << "\n";
            return 1;
        }
        if (!game) {
            std::cerr << "Lowering produced no runtime game.\n";
            return 1;
        }
        std::cout << serializeRuntimeGameDebugJson(*game);
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
    const int result = stepCommandForGame(game, inputTokens);
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
    const int result = diffTraceAgainstSnapshots(game, loadTraceSnapshotsFromJsonText(runTraceExporterAndCaptureJson(sourcePath, traceArgs)), std::cerr, true);
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
                ps_session* session = nullptr;
                ps_error* error = nullptr;
                const auto sessionStart = std::chrono::steady_clock::now();
                if (!ps_session_create(game, &session, &error)) {
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
                char* serialized = ps_session_serialize_test_string(session);
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
                ps_session_destroy(session);

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
        << "  puzzlescript_cpp test js-parity <generated-js-parity-data.json>\n"
        << "      Check saved replay cases generated from the original JavaScript test suite.\n"
        << "  puzzlescript_cpp bench game.txt --iterations 10000 --threads 4\n"
        << "      Benchmark clone/hash/session operations for a source game.\n\n"
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
        << "  puzzlescript_cpp help test\n"
        << "  puzzlescript_cpp help profile\n"
        << "  puzzlescript_cpp help bench\n";
}

void printPlayHelp() {
    std::cout
        << "Usage: puzzlescript_cpp play game.txt [--level N] [--seed seed] [--settle-again] [--native-compile]\n\n"
        << "Opens the SDL player for a PuzzleScript source file. The runtime currently loads\n"
        << "through generated JS parity data while the native compiler is being connected to\n"
        << "full Game lowering.\n\n"
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

void printTestHelp() {
    std::cout
        << "Usage: puzzlescript_cpp test js-parity generated-js-parity-data.json [options]\n"
        << "       puzzlescript_cpp test diagnostics parser-corpus.bundle.ndjson\n\n"
        << "JS parity corpus means saved replay and diagnostic cases generated from the\n"
        << "original JavaScript test suite: testdata.js and errormessage_testdata.js.\n"
        << "Simulation tests compare gameplay traces. Compiler tests compare diagnostics.\n\n"
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
        << "Usage: puzzlescript_cpp profile-simulations generated-js-parity-data.json [--repeat N] [--profile-timers] [--quiet]\n\n"
        << "Runs the C++ runtime over the saved simulation replay corpus without invoking\n"
        << "the JavaScript engine or doing JS-vs-C++ parity comparison. This measures the\n"
        << "compiled runtime representation loaded from corpus IR, not native source\n"
        << "parse/compile time. Use --repeat to amplify engine stepping time for sampling\n"
        << "profilers.\n\n"
        << "Usually use:\n"
        << "  make profile_simulation_tests\n";
}

void printBenchHelp() {
    std::cout
        << "Usage: puzzlescript_cpp bench game.txt [--level N] [--seed seed] [--settle-again] [--native-compile] [--iterations N] [--threads N]\n\n"
        << "Benchmarks native runtime operations for a source game. Use --threads to measure\n"
        << "multi-session throughput for future solver workloads.\n\n"
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
        if (command == "diagnostics-parity") {
            return diagnosticsParityMain(std::filesystem::path(path));
        }
        if (command == "test") {
            if (path == "js-parity" && argc >= 4) {
                return checkTraceSweepCommand(argv[3], argc - 4, argv + 4);
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
