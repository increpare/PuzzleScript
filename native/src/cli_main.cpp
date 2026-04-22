#include <array>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <memory>
#include <optional>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#include "json.hpp"
#include "puzzlescript/puzzlescript.h"

#ifdef PS_HAVE_SDL2
int ps_cli_run_player(const std::string& irPath);
#endif

namespace {

#ifndef PS_NODE_EXECUTABLE
#define PS_NODE_EXECUTABLE "node"
#endif

#ifndef PS_EXPORT_IR_SCRIPT
#define PS_EXPORT_IR_SCRIPT "src/tests/export_ir_json.js"
#endif

#ifndef PS_EXPORT_TRACE_SCRIPT
#define PS_EXPORT_TRACE_SCRIPT "src/tests/export_execution_trace.js"
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

int32_t looseAsInt(const puzzlescript::json::Value& value) {
    if (value.isInteger()) {
        return static_cast<int32_t>(value.asInteger());
    }
    if (value.isDouble()) {
        return static_cast<int32_t>(value.asDouble());
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
        if (item.isInteger() || item.isDouble()) {
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
    for (const auto& snapshotValue : snapshotsValue->asArray()) {
        const auto& object = snapshotValue.asObject();
        TraceSnapshot snapshot;
        snapshot.phase = object.at("phase").asString();
        if (const auto inputIt = object.find("input"); inputIt != object.end() && !inputIt->second.isNull()) {
            if (inputIt->second.isInteger() || inputIt->second.isDouble()) {
                snapshot.numericInput = looseAsInt(inputIt->second);
            } else if (inputIt->second.isString()) {
                snapshot.stringInput = inputIt->second.asString();
            }
        }
        snapshot.currentLevelIndex = looseAsInt(object.at("current_level_index"));
        if (const auto targetIt = object.find("current_level_target"); targetIt != object.end() && !targetIt->second.isNull()) {
            snapshot.currentLevelTarget = looseAsInt(targetIt->second);
        }
        if (const auto loadedLevelSeedIt = object.find("loaded_level_seed"); loadedLevelSeedIt != object.end() && !loadedLevelSeedIt->second.isNull()) {
            snapshot.loadedLevelSeed = loadedLevelSeedIt->second.asString();
        }
        snapshot.titleScreen = object.at("title_screen").asBool();
        snapshot.textMode = object.at("text_mode").asBool();
        if (const auto titleModeIt = object.find("title_mode"); titleModeIt != object.end()) {
            snapshot.titleMode = looseAsInt(titleModeIt->second);
        }
        if (const auto titleSelectionIt = object.find("title_selection"); titleSelectionIt != object.end()) {
            snapshot.titleSelection = looseAsInt(titleSelectionIt->second);
        }
        if (const auto titleSelectedIt = object.find("title_selected"); titleSelectedIt != object.end()) {
            snapshot.titleSelected = titleSelectedIt->second.asBool();
        }
        if (const auto messageSelectedIt = object.find("message_selected"); messageSelectedIt != object.end()) {
            snapshot.messageSelected = messageSelectedIt->second.asBool();
        }
        snapshot.winning = object.at("winning").asBool();
        if (const auto randomStateValidIt = object.find("random_state_valid"); randomStateValidIt != object.end()) {
            snapshot.randomStateValid = randomStateValidIt->second.asBool();
        }
        if (const auto randomStateIIt = object.find("random_state_i"); randomStateIIt != object.end()) {
            snapshot.randomStateI = looseAsInt(randomStateIIt->second);
        }
        if (const auto randomStateJIt = object.find("random_state_j"); randomStateJIt != object.end()) {
            snapshot.randomStateJ = looseAsInt(randomStateJIt->second);
        }
        if (const auto previewBytesIt = object.find("random_state_preview_bytes"); previewBytesIt != object.end()) {
            snapshot.randomStatePreviewBytes = looseAsIntArray(previewBytesIt->second);
        }
        snapshot.serializedLevel = object.at("serialized_level").asString();
        if (const auto soundsIt = object.find("new_sounds"); soundsIt != object.end() && soundsIt->second.isArray()) {
            for (const auto& soundValue : soundsIt->second.asArray()) {
                if (soundValue.isInteger()) {
                    snapshot.newSounds.push_back(static_cast<int32_t>(soundValue.asInteger()));
                } else if (soundValue.isDouble()) {
                    snapshot.newSounds.push_back(static_cast<int32_t>(soundValue.asDouble()));
                } else if (soundValue.isString()) {
                    snapshot.newSounds.push_back(std::stoi(soundValue.asString()));
                }
            }
        }
        snapshots.push_back(std::move(snapshot));
    }

    return snapshots;
}

std::vector<TraceSnapshot> loadTraceSnapshots(const std::filesystem::path& path) {
    return loadTraceSnapshotsFromJsonText(readFile(path));
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

bool loadGameFromFile(const std::filesystem::path& path, ps_game** outGame) {
    return loadGameFromJsonText(readFile(path), outGame);
}

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
    const auto actualSnapshotValue = puzzlescript::json::parse(actualSnapshotJson);
    const auto& actualSnapshotObject = actualSnapshotValue.asObject();
    std::optional<std::string> actualLoadedLevelSeed;
    if (const auto loadedLevelSeedIt = actualSnapshotObject.find("loaded_level_seed"); loadedLevelSeedIt != actualSnapshotObject.end() && !loadedLevelSeedIt->second.isNull()) {
        actualLoadedLevelSeed = loadedLevelSeedIt->second.asString();
    }

    bool ok = true;
    if (actualSerialized != expected.serializedLevel) {
        stream << "snapshot[" << snapshotIndex << "] serialized level mismatch\n";
        stream << "expected_serialized_level:\n" << expected.serializedLevel;
        if (!expected.serializedLevel.empty() && expected.serializedLevel.back() != '\n') {
            stream << "\n";
        }
        stream << "actual_serialized_level:\n" << actualSerialized;
        if (!actualSerialized.empty() && actualSerialized.back() != '\n') {
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
        && actualLoadedLevelSeed.has_value()
        && *expected.loadedLevelSeed != *actualLoadedLevelSeed;
    if (!skipRandomStateComparison) {
        const bool actualRandomStateValid = actualSnapshotObject.at("random_state_valid").asBool();
        if (actualRandomStateValid != expected.randomStateValid) {
            stream << "snapshot[" << snapshotIndex << "] random_state_valid mismatch: actual="
                   << (actualRandomStateValid ? 1 : 0) << " expected=" << (expected.randomStateValid ? 1 : 0) << "\n";
            ok = false;
        }
        const int32_t actualRandomStateI = looseAsInt(actualSnapshotObject.at("random_state_i"));
        if (actualRandomStateI != expected.randomStateI) {
            stream << "snapshot[" << snapshotIndex << "] random_state_i mismatch: actual="
                   << actualRandomStateI << " expected=" << expected.randomStateI << "\n";
            ok = false;
        }
        const int32_t actualRandomStateJ = looseAsInt(actualSnapshotObject.at("random_state_j"));
        if (actualRandomStateJ != expected.randomStateJ) {
            stream << "snapshot[" << snapshotIndex << "] random_state_j mismatch: actual="
                   << actualRandomStateJ << " expected=" << expected.randomStateJ << "\n";
            ok = false;
        }
        std::vector<int32_t> actualPreviewBytes;
        if (const auto previewIt = actualSnapshotObject.find("random_state_preview_bytes"); previewIt != actualSnapshotObject.end()) {
            actualPreviewBytes = looseAsIntArray(previewIt->second);
        }
        if (actualPreviewBytes != expected.randomStatePreviewBytes) {
            stream << "snapshot[" << snapshotIndex << "] random_state_preview_bytes mismatch: actual=[";
            appendIntList(stream, actualPreviewBytes);
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

int runSourceCommand(const std::string& sourcePath, int argc, char** argv) {
    std::vector<std::string> exporterArgs;
    for (int index = 0; index < argc; ++index) {
        exporterArgs.emplace_back(argv[index]);
    }

    ps_game* game = nullptr;
    if (!loadGameFromJsonText(runIrExporterAndCaptureJson(sourcePath, exporterArgs), &game)) {
        return 1;
    }
    const int result = runCommandForGame(game);
    ps_free_game(game);
    return result;
}

int benchSourceCommand(const std::string& sourcePath, int argc, char** argv) {
    uint32_t iterations = 10000;
    uint32_t threads = 1;
    std::vector<std::string> exporterArgs;

    for (int index = 0; index < argc; ++index) {
        const std::string arg = argv[index];
        if (arg == "--iterations" && index + 1 < argc) {
            iterations = static_cast<uint32_t>(std::stoul(argv[++index]));
        } else if (arg == "--threads" && index + 1 < argc) {
            threads = static_cast<uint32_t>(std::stoul(argv[++index]));
        } else {
            exporterArgs.emplace_back(arg);
        }
    }

    ps_game* game = nullptr;
    if (!loadGameFromJsonText(runIrExporterAndCaptureJson(sourcePath, exporterArgs), &game)) {
        return 1;
    }
    const int result = benchCommandForGame(game, iterations, threads);
    ps_free_game(game);
    return result;
}

int stepSourceCommand(const std::string& sourcePath, int argc, char** argv) {
    std::vector<std::string> exporterArgs;
    std::vector<std::string> inputTokens;

    for (int index = 0; index < argc; ++index) {
        const std::string arg = argv[index];
        if ((arg == "--level" || arg == "--seed") && index + 1 < argc) {
            exporterArgs.push_back(arg);
            exporterArgs.push_back(argv[++index]);
        } else if (arg == "--settle-again") {
            exporterArgs.push_back(arg);
        } else {
            inputTokens.push_back(arg);
        }
    }

    ps_game* game = nullptr;
    if (!loadGameFromJsonText(runIrExporterAndCaptureJson(sourcePath, exporterArgs), &game)) {
        return 1;
    }
    const int result = stepCommandForGame(game, inputTokens);
    ps_free_game(game);
    return result;
}

int diffTraceSourceCommand(const std::string& sourcePath, int argc, char** argv) {
    std::vector<std::string> irArgs;
    std::vector<std::string> traceArgs;

    for (int index = 0; index < argc; ++index) {
        const std::string arg = argv[index];
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
    if (!loadGameFromJsonText(runIrExporterAndCaptureJson(sourcePath, irArgs), &game)) {
        return 1;
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
        } else {
            throw std::runtime_error("Unsupported test-fixtures argument: " + arg);
        }
    }

    try {
        const auto manifestValue = puzzlescript::json::parse(readFile(manifestPath));
        const auto* fixturesValue = manifestValue.find("simulation_fixtures");
        if (!fixturesValue || !fixturesValue->isArray()) {
            throw std::runtime_error("fixtures.json is missing simulation_fixtures");
        }

        const auto& fixtures = fixturesValue->asArray();
        for (size_t index = 0; index < fixtures.size(); ++index) {
            try {
                const auto& fixture = fixtures[index].asObject();
                const auto irPath = manifestDir / fixture.at("ir_file").asString();
                const auto name = fixture.at("name").asString();
                const auto expectedSerialized = fixture.at("initial_serialized_level").asString();

                ps_game* game = nullptr;
                if (!loadGameFromFile(irPath, &game)) {
                    ++preparedFailed;
                    continue;
                }
                ps_session* session = nullptr;
                ps_error* error = nullptr;
                if (!ps_session_create(game, &session, &error)) {
                    std::cerr << name << ": " << ps_error_message(error) << "\n";
                    ps_free_error(error);
                    ps_free_game(game);
                    ++preparedFailed;
                    continue;
                }

                char* serialized = ps_session_serialize_test_string(session);
                const std::string actual = serialized ? serialized : "";
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

                const bool shouldCheckTrace = fixture.find("trace_file") != fixture.end()
                    && (traceAll || (traceLimit > 0 && traceChecked < traceLimit));
                if (shouldCheckTrace) {
                    ++traceChecked;
                    if (traceProgress > 0 && (traceChecked % traceProgress) == 0) {
                        std::cerr << "trace_progress checked=" << traceChecked
                                  << " passed=" << tracePassed
                                  << " failed=" << traceFailed
                                  << " current_fixture=" << name
                                  << "\n";
                    }
                    std::ostringstream traceErrors;
                    const auto tracePath = manifestDir / fixture.at("trace_file").asString();
                    ps_game* traceGame = nullptr;
                    if (!loadGameFromFile(irPath, &traceGame)) {
                        ++traceFailed;
                        if (!traceQuiet) {
                            std::cerr << name << ": failed to reload game for trace replay\n";
                        }
                    } else {
                        const int traceResult = diffTraceAgainstSnapshots(traceGame, loadTraceSnapshots(tracePath), traceErrors, false);
                        ps_free_game(traceGame);
                        if (traceResult == 0) {
                            ++tracePassed;
                        } else {
                            ++traceFailed;
                            if (!traceQuiet) {
                                std::cerr << name << ": trace replay mismatch\n" << traceErrors.str();
                            }
                        }
                    }
                }

                ps_free_game(game);
            } catch (const std::exception& error) {
                ++preparedFailed;
                if (!traceQuiet) {
                    std::cerr << "fixture #" << index << ": " << error.what() << "\n";
                }
            }
        }
    } catch (const std::exception& error) {
        std::cerr << error.what() << "\n";
        return 1;
    }

    std::cout << "simulation_fixture_count=" << (preparedPassed + preparedFailed)
              << " prepared_session_checks_passed=" << preparedPassed
              << " prepared_session_checks_failed=" << preparedFailed
              << " trace_replay_checked=" << traceChecked
              << " trace_replay_passed=" << tracePassed
              << " trace_replay_failed=" << traceFailed
              << "\n";
    const bool preparedOk = preparedFailed == 0;
    const bool traceOk = traceAllowFailures || traceFailed == 0;
    return (preparedOk && traceOk) ? 0 : 1;
}

void printUsage() {
    std::cerr << "Usage:\n"
              << "  ps_cli run <ir.json>\n"
              << "  ps_cli step <ir.json> [input ...]\n"
              << "  ps_cli bench <ir.json> [--iterations N] [--threads N]\n"
              << "  ps_cli run-source <game.ps> [--level N] [--seed seed] [--settle-again]\n"
              << "  ps_cli step-source <game.ps> [input ...] [--level N] [--seed seed] [--settle-again]\n"
              << "  ps_cli bench-source <game.ps> [--level N] [--seed seed] [--settle-again] [--iterations N] [--threads N]\n"
              << "  ps_cli diff-trace <ir.json> <trace.json>\n"
              << "  ps_cli trace-at <ir.json> <trace.json> <snapshot-index>\n"
              << "  ps_cli trace-step-at <ir.json> <trace.json> <snapshot-index>\n"
              << "  ps_cli diff-trace-source <game.ps> [--level N] [--seed seed] [--inputs-json json] [--inputs-file path]\n"
              << "  ps_cli test-fixtures <fixtures.json> [--trace-limit N] [--trace-all] [--trace-allow-failures] [--trace-quiet] [--trace-progress N]\n"
              << "  ps_cli play <ir.json>\n";
}

} // namespace

int main(int argc, char** argv) {
    if (argc < 3) {
        printUsage();
        return 1;
    }

    const std::string command = argv[1];
    const std::string path = argv[2];

    try {
        if (command == "run") {
            return runCommand(path, argc - 3, argv + 3);
        }
        if (command == "step") {
            return stepCommand(path, argc - 3, argv + 3);
        }
        if (command == "bench") {
            return benchCommand(path, argc - 3, argv + 3);
        }
        if (command == "run-source") {
            return runSourceCommand(path, argc - 3, argv + 3);
        }
        if (command == "step-source") {
            return stepSourceCommand(path, argc - 3, argv + 3);
        }
        if (command == "bench-source") {
            return benchSourceCommand(path, argc - 3, argv + 3);
        }
        if (command == "diff-trace") {
            if (argc < 4) {
                printUsage();
                return 1;
            }
            return diffTraceCommand(path, argv[3]);
        }
        if (command == "trace-at") {
            if (argc < 5) {
                printUsage();
                return 1;
            }
            return traceAtCommand(path, argv[3], static_cast<size_t>(std::stoull(argv[4])));
        }
        if (command == "trace-step-at") {
            if (argc < 5) {
                printUsage();
                return 1;
            }
            return traceStepAtCommand(path, argv[3], static_cast<size_t>(std::stoull(argv[4])));
        }
        if (command == "diff-trace-source") {
            return diffTraceSourceCommand(path, argc - 3, argv + 3);
        }
        if (command == "test-fixtures") {
            return testFixturesCommand(path, argc - 3, argv + 3);
        }
        if (command == "play") {
#ifdef PS_HAVE_SDL2
            return ps_cli_run_player(path);
#else
            std::cerr << "SDL2 support is not enabled in this build.\n";
            return 1;
#endif
        }
    } catch (const std::exception& error) {
        std::cerr << error.what() << "\n";
        return 1;
    }

    printUsage();
    return 1;
}
