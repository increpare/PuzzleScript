#include "runtime/core.hpp"
#include "runtime/compiled_rules.hpp"

#include <algorithm>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <vector>

 #include "compiler/lower_to_runtime.hpp"
 #include "compiler/parser.hpp"

using puzzlescript::CompileResult;
using puzzlescript::SpecializedCompactTurnOutcome;
using puzzlescript::SpecializedCompactTurnBackend;
using puzzlescript::CompactStateView;
using puzzlescript::Error;
using puzzlescript::FullState;
using puzzlescript::Game;
using puzzlescript::kMaskWordBits;
using puzzlescript::MaskWordUnsigned;
using puzzlescript::RuntimeStepOptions;

struct ps_game {
    std::shared_ptr<const Game> impl;
};

struct ps_full_state {
    std::unique_ptr<FullState> impl;
};

struct ps_compile_result {
    std::unique_ptr<CompileResult> impl;
};

struct ps_error {
    std::unique_ptr<Error> impl;
};

namespace {

ps_error* makeError(std::unique_ptr<Error> error) {
    if (!error) {
        return nullptr;
    }
    auto* wrapper = new ps_error();
    wrapper->impl = std::move(error);
    return wrapper;
}

char* duplicateString(const std::string& value) {
    char* buffer = new char[value.size() + 1];
    std::memcpy(buffer, value.c_str(), value.size() + 1);
    return buffer;
}

struct CompactOracleState {
    std::vector<uint64_t> objectBits;
    std::vector<puzzlescript::MaskWord> movementWords;
    FullState::RandomState randomState;
};

uint32_t compactOracleTrailingZeros(MaskWordUnsigned value) {
    if constexpr (sizeof(MaskWordUnsigned) <= sizeof(unsigned int)) {
        return static_cast<uint32_t>(__builtin_ctz(static_cast<unsigned int>(value)));
    } else {
        return static_cast<uint32_t>(__builtin_ctzll(static_cast<unsigned long long>(value)));
    }
}

CompactOracleState compactOracleStateFromFullState(const FullState& session) {
    CompactOracleState state;
    state.randomState = session.randomState;
    state.movementWords = session.liveMovements;
    const int32_t tileCount = session.liveLevel.width * session.liveLevel.height;
    const size_t cellWordCount = static_cast<size_t>((tileCount + 63) / 64);
    const int32_t objectCount = session.game ? session.game->objectCount : 0;
    state.objectBits.assign(static_cast<size_t>(std::max(objectCount, 0)) * cellWordCount, 0);
    if (objectCount <= 0 || tileCount <= 0 || cellWordCount == 0) {
        return state;
    }
    const int32_t stride = session.game->strideObject;
    for (int32_t tileIndex = 0; tileIndex < tileCount; ++tileIndex) {
        const size_t sourceBase = static_cast<size_t>(tileIndex * stride);
        const size_t bitWord = static_cast<size_t>(tileIndex >> 6);
        const uint64_t bitMask = uint64_t{1} << static_cast<uint32_t>(tileIndex & 63);
        for (int32_t word = 0; word < stride; ++word) {
            MaskWordUnsigned bits = static_cast<MaskWordUnsigned>(session.liveLevel.objects[sourceBase + static_cast<size_t>(word)]);
            while (bits != 0) {
                const uint32_t bit = compactOracleTrailingZeros(bits);
                const int32_t objectId = word * static_cast<int32_t>(kMaskWordBits) + static_cast<int32_t>(bit);
                if (objectId < objectCount) {
                    state.objectBits[static_cast<size_t>(objectId) * cellWordCount + bitWord] |= bitMask;
                }
                bits &= bits - 1;
            }
        }
    }
    return state;
}

bool compactOracleStatesEqual(const CompactOracleState& lhs, const CompactOracleState& rhs) {
    return lhs.objectBits == rhs.objectBits
        && lhs.movementWords == rhs.movementWords
        && lhs.randomState.s == rhs.randomState.s
        && lhs.randomState.i == rhs.randomState.i
        && lhs.randomState.j == rhs.randomState.j
        && lhs.randomState.valid == rhs.randomState.valid;
}

void debugCompactOracleStateMismatch(const CompactOracleState& compact, const CompactOracleState& interpreter) {
    if (std::getenv("PS_COMPACT_ORACLE_DEBUG") == nullptr) {
        return;
    }
    if (compact.objectBits != interpreter.objectBits) {
        const size_t count = std::min(compact.objectBits.size(), interpreter.objectBits.size());
        for (size_t index = 0; index < count; ++index) {
            if (compact.objectBits[index] != interpreter.objectBits[index]) {
                std::cerr << "compact oracle objectBits mismatch index=" << index
                          << " compact=" << compact.objectBits[index]
                          << " interpreter=" << interpreter.objectBits[index]
                          << "\n";
                return;
            }
        }
        std::cerr << "compact oracle objectBits size mismatch compact=" << compact.objectBits.size()
                  << " interpreter=" << interpreter.objectBits.size()
                  << "\n";
        return;
    }
    if (compact.movementWords != interpreter.movementWords) {
        const size_t count = std::min(compact.movementWords.size(), interpreter.movementWords.size());
        for (size_t index = 0; index < count; ++index) {
            if (compact.movementWords[index] != interpreter.movementWords[index]) {
                std::cerr << "compact oracle movementWords mismatch index=" << index
                          << " compact=" << compact.movementWords[index]
                          << " interpreter=" << interpreter.movementWords[index]
                          << "\n";
                return;
            }
        }
        std::cerr << "compact oracle movementWords size mismatch compact=" << compact.movementWords.size()
                  << " interpreter=" << interpreter.movementWords.size()
                  << "\n";
        return;
    }
    std::cerr << "compact oracle random state mismatch\n";
}

bool equivalentCompactOracleStepResult(const ps_step_result& lhs, const ps_step_result& rhs) {
    const bool terminal = lhs.won || rhs.won || lhs.restarted || rhs.restarted || lhs.transitioned || rhs.transitioned;
    return lhs.changed == rhs.changed
        && lhs.won == rhs.won
        && lhs.restarted == rhs.restarted
        && (terminal || lhs.transitioned == rhs.transitioned);
}

} // namespace

bool ps_load_ir_json(const char* json_utf8, size_t json_size, ps_game** out_game, ps_error** out_error) {
    if (out_error) {
        *out_error = nullptr;
    }
    if (!json_utf8 || !out_game) {
        if (out_error) {
            *out_error = makeError(std::make_unique<Error>("ps_load_ir_json received null input"));
        }
        return false;
    }

    std::shared_ptr<const Game> game;
    if (auto error = puzzlescript::loadGameFromJson(std::string_view(json_utf8, json_size), game)) {
        if (out_error) {
            *out_error = makeError(std::move(error));
        }
        return false;
    }

    auto* wrapper = new ps_game();
    wrapper->impl = std::move(game);
    *out_game = wrapper;
    return true;
}

bool ps_compile_source(const char* source_utf8, size_t source_size, ps_compile_result** out_result) {
    if (!out_result) {
        return false;
    }
    auto* wrapper = new ps_compile_result();
    wrapper->impl = std::make_unique<CompileResult>();
    try {
        puzzlescript::compiler::DiagnosticSink diagnostics;
        const auto state = puzzlescript::compiler::parseSource(
            source_utf8 == nullptr ? std::string_view{} : std::string_view(source_utf8, source_size),
            diagnostics
        );
        // For now, treat any lowering failure as a compile error. (Once lowering
        // is implemented, we can choose to gate on diagnostic severity.)
        std::shared_ptr<const Game> game;
        if (auto error = puzzlescript::compiler::lowerToRuntimeGame(state, game)) {
            wrapper->impl->error = std::move(error);
            *out_result = wrapper;
            return false;
        }
        if (game) {
            puzzlescript::attachLinkedCompiledRules(
                *std::const_pointer_cast<Game>(game),
                source_utf8 == nullptr ? std::string_view{} : std::string_view(source_utf8, source_size)
            );
        }
        wrapper->impl->game = std::move(game);
        *out_result = wrapper;
        return true;
    } catch (const std::exception& e) {
        wrapper->impl->error = std::make_unique<Error>(e.what());
        *out_result = wrapper;
        return false;
    }
}

const ps_game* ps_compile_result_game(const ps_compile_result* result) {
    if (!result || !result->impl || !result->impl->game) {
        return nullptr;
    }
    auto* wrapper = new ps_game();
    wrapper->impl = result->impl->game;
    return wrapper;
}

const ps_error* ps_compile_result_error(const ps_compile_result* result) {
    if (!result || !result->impl || !result->impl->error) {
        return nullptr;
    }
    auto* wrapper = new ps_error();
    wrapper->impl = std::make_unique<Error>(result->impl->error->message);
    return wrapper;
}

void ps_free_compile_result(ps_compile_result* result) {
    delete result;
}

ps_step_result ps_full_state_turn(ps_full_state* state, ps_input input) {
    if (!state) {
        return ps_step_result{};
    }
    return puzzlescript::turn(*state->impl, input, RuntimeStepOptions{});
}

bool ps_full_state_create(const ps_game* game, ps_full_state** out_state, ps_error** out_error) {
    if (out_error) {
        *out_error = nullptr;
    }
    if (!game || !out_state) {
        if (out_error) {
            *out_error = makeError(std::make_unique<Error>("ps_full_state_create received null input"));
        }
        return false;
    }
    auto* wrapper = new ps_full_state();
    wrapper->impl = puzzlescript::createFullState(game->impl);
    *out_state = wrapper;
    return true;
}

bool ps_full_state_create_with_loaded_level_seed(
    const ps_game* game,
    const char* loaded_level_seed_utf8,
    ps_full_state** out_state,
    ps_error** out_error
) {
    if (out_error) {
        *out_error = nullptr;
    }
    if (!game || !out_state) {
        if (out_error) {
            *out_error = makeError(std::make_unique<Error>("ps_full_state_create_with_loaded_level_seed received null input"));
        }
        return false;
    }
    if (!loaded_level_seed_utf8) {
        return ps_full_state_create(game, out_state, out_error);
    }
    auto* wrapper = new ps_full_state();
    wrapper->impl = puzzlescript::createFullStateWithLoadedLevelSeed(game->impl, loaded_level_seed_utf8);
    *out_state = wrapper;
    return true;
}

bool ps_full_state_clone(const ps_full_state* state, ps_full_state** out_state, ps_error** out_error) {
    if (out_error) {
        *out_error = nullptr;
    }
    if (!state || !out_state) {
        if (out_error) {
            *out_error = makeError(std::make_unique<Error>("ps_full_state_clone received null input"));
        }
        return false;
    }
    auto* wrapper = new ps_full_state();
    wrapper->impl = std::make_unique<FullState>(*state->impl);
    *out_state = wrapper;
    return true;
}

void ps_full_state_destroy(ps_full_state* state) {
    delete state;
}

void ps_full_state_set_unit_testing(ps_full_state* state, bool enabled) {
    if (state == nullptr || !state->impl) {
        return;
    }
    state->impl->suppressRuleMessages = enabled;
}

bool ps_full_state_load_level(ps_full_state* state, int32_t level_index, ps_error** out_error) {
    if (out_error) {
        *out_error = nullptr;
    }
    if (!state) {
        if (out_error) {
            *out_error = makeError(std::make_unique<Error>("ps_full_state_load_level received null state"));
        }
        return false;
    }
    if (auto error = puzzlescript::loadLevel(*state->impl, level_index)) {
        if (out_error) {
            *out_error = makeError(std::move(error));
        }
        return false;
    }
    return true;
}

bool ps_full_state_compact_turn_oracle_check(
    const ps_full_state* state,
    ps_input input,
    ps_compact_turn_oracle_info* out_info
) {
    if (out_info) {
        *out_info = ps_compact_turn_oracle_info{};
        out_info->matched = true;
    }
    if (state == nullptr || !state->impl || !state->impl->game) {
        return false;
    }
    const FullState& original = *state->impl;
    if (original.preparedFullState.titleScreen || original.preparedFullState.textMode) {
        return true;
    }
    const SpecializedCompactTurnBackend* backend = original.game->specializedCompactTurn;
    if (backend == nullptr || backend->step == nullptr || !backend->support.wholeTurnSupported) {
        return true;
    }

    CompactOracleState compact = compactOracleStateFromFullState(original);
    CompactStateView view{
        compact.objectBits.empty() ? nullptr : compact.objectBits.data(),
        compact.objectBits.size(),
        compact.movementWords.empty() ? nullptr : compact.movementWords.data(),
        compact.movementWords.size(),
        original.liveLevel.width,
        original.liveLevel.height,
        compact.randomState.s.data(),
        compact.randomState.s.size(),
        &compact.randomState.i,
        &compact.randomState.j,
        &compact.randomState.valid,
        original.preparedFullState.currentLevelIndex,
    };
    RuntimeStepOptions options{};
    options.emitAudio = false;
    options.againPolicy = puzzlescript::AgainPolicy::Drain;
    const SpecializedCompactTurnOutcome compactOutcome = backend->step(*original.game, view, input, options);

    FullState interpreter = original;
    ps_step_result interpreterResult = interpretedTurn(interpreter, input, options);

    bool matched = compactOutcome.handled
        && equivalentCompactOracleStepResult(compactOutcome.result, interpreterResult);
    bool stateChecked = false;
    const bool terminal = compactOutcome.result.won
        || interpreterResult.won
        || compactOutcome.result.restarted
        || interpreterResult.restarted
        || compactOutcome.result.transitioned
        || interpreterResult.transitioned;
    if (matched
        && !terminal
        && interpreter.liveLevel.width == original.liveLevel.width
        && interpreter.liveLevel.height == original.liveLevel.height) {
        stateChecked = true;
        const CompactOracleState interpreterState = compactOracleStateFromFullState(interpreter);
        matched = compactOracleStatesEqual(compact, interpreterState);
        if (!matched) {
            debugCompactOracleStateMismatch(compact, interpreterState);
        }
    }

    if (out_info) {
        out_info->attempted = true;
        out_info->handled = compactOutcome.handled;
        out_info->matched = matched;
        out_info->state_checked = stateChecked;
        out_info->compact_result = compactOutcome.result;
        out_info->interpreter_result = interpreterResult;
    }
    return true;
}

bool ps_full_state_pending_again(const ps_full_state* state) {
    return state && state->impl->pendingAgain;
}

bool ps_full_state_undo(ps_full_state* state) {
    return state ? puzzlescript::undo(*state->impl) : false;
}

bool ps_full_state_restart(ps_full_state* state) {
    return state ? puzzlescript::restart(*state->impl) : false;
}

bool ps_full_state_advance_level(ps_full_state* state, ps_error** out_error) {
    if (out_error) {
        *out_error = nullptr;
    }
    if (!state) {
        if (out_error) {
            *out_error = makeError(std::make_unique<Error>("ps_full_state_advance_level received null state"));
        }
        return false;
    }
    if (auto error = puzzlescript::advanceLevel(*state->impl)) {
        if (out_error) {
            *out_error = makeError(std::move(error));
        }
        return false;
    }
    return true;
}

void ps_full_state_status(const ps_full_state* state, ps_full_state_status_info* out_status) {
    if (!state || !out_status) {
        return;
    }
    out_status->mode = state->impl->preparedFullState.titleScreen
        ? PS_FULL_STATE_MODE_TITLE
        : (state->impl->preparedFullState.textMode ? PS_FULL_STATE_MODE_MESSAGE : PS_FULL_STATE_MODE_LEVEL);
    out_status->current_level_index = state->impl->preparedFullState.currentLevelIndex;
    out_status->has_current_level_target = state->impl->preparedFullState.currentLevelTarget.has_value();
    out_status->current_level_target = state->impl->preparedFullState.currentLevelTarget.value_or(0);
    out_status->width = state->impl->liveLevel.width;
    out_status->height = state->impl->liveLevel.height;
    out_status->title_mode = state->impl->preparedFullState.titleMode;
    out_status->title_selection = state->impl->preparedFullState.titleSelection;
    out_status->can_undo = state->impl->canUndo;
    out_status->winning = state->impl->preparedFullState.winning;
    out_status->title_screen = state->impl->preparedFullState.titleScreen;
    out_status->text_mode = state->impl->preparedFullState.textMode;
    out_status->title_selected = state->impl->preparedFullState.titleSelected;
    out_status->message_selected = state->impl->preparedFullState.messageSelected;
}

const char* ps_full_state_message_text(const ps_full_state* state) {
    if (!state || !state->impl) {
        return "";
    }
    const auto& prepared = state->impl->preparedFullState;
    if (!prepared.messageText.empty()) {
        return prepared.messageText.c_str();
    }
    if (prepared.textMode && prepared.level.isMessage) {
        return prepared.level.message.c_str();
    }
    return "";
}

bool ps_full_state_cell_has_object(const ps_full_state* state, int32_t x, int32_t y, int32_t object_id) {
    if (!state || !state->impl || object_id < 0) {
        return false;
    }
    const FullState& impl = *state->impl;
    if (x < 0 || y < 0 || x >= impl.liveLevel.width || y >= impl.liveLevel.height) {
        return false;
    }
    if (object_id >= impl.game->objectCount) {
        return false;
    }
    const uint32_t word = puzzlescript::maskWordIndex(static_cast<uint32_t>(object_id));
    if (word >= impl.game->wordCount) {
        return false;
    }
    const int32_t tile_index = x * impl.liveLevel.height + y;
    const size_t offset = static_cast<size_t>(tile_index) * impl.game->wordCount + word;
    if (offset >= impl.liveLevel.objects.size()) {
        return false;
    }
    return (impl.liveLevel.objects[offset] & puzzlescript::maskBit(static_cast<uint32_t>(object_id))) != 0;
}

bool ps_full_state_first_player_position(const ps_full_state* state, int32_t* out_x, int32_t* out_y) {
    if (out_x) {
        *out_x = 0;
    }
    if (out_y) {
        *out_y = 0;
    }
    if (!state || !state->impl) {
        return false;
    }
    const FullState& impl = *state->impl;
    if (impl.game->playerMask == puzzlescript::kNullMaskOffset || impl.liveLevel.width <= 0 || impl.liveLevel.height <= 0) {
        return false;
    }
    const puzzlescript::MaskWord* playerMask = impl.game->maskArena.data() + impl.game->playerMask;
    const int32_t tileCount = impl.liveLevel.width * impl.liveLevel.height;
    for (int32_t tile_index = 0; tile_index < tileCount; ++tile_index) {
        const size_t cellBase = static_cast<size_t>(tile_index) * impl.game->wordCount;
        bool containsPlayer = impl.game->playerMaskAggregate;
        if (impl.game->playerMaskAggregate) {
            for (uint32_t word = 0; word < impl.game->wordCount; ++word) {
                if ((impl.liveLevel.objects[cellBase + word] & playerMask[word]) != playerMask[word]) {
                    containsPlayer = false;
                    break;
                }
            }
        } else {
            containsPlayer = false;
            for (uint32_t word = 0; word < impl.game->wordCount; ++word) {
                if ((impl.liveLevel.objects[cellBase + word] & playerMask[word]) != 0) {
                    containsPlayer = true;
                    break;
                }
            }
        }
        if (!containsPlayer) {
            continue;
        }
        if (out_x) {
            *out_x = tile_index / impl.liveLevel.height;
        }
        if (out_y) {
            *out_y = tile_index % impl.liveLevel.height;
        }
        return true;
    }
    return false;
}

uint64_t ps_full_state_hash64(const ps_full_state* state) {
    return state ? puzzlescript::hashFullState64(*state->impl) : 0;
}

ps_hash128 ps_full_state_hash128(const ps_full_state* state) {
    return state ? puzzlescript::hashFullState128(*state->impl) : ps_hash128{};
}

char* ps_full_state_serialize_test_string(const ps_full_state* state) {
    if (!state) {
        return nullptr;
    }
    return duplicateString(puzzlescript::serializeTestString(*state->impl));
}

char* ps_full_state_export_snapshot(const ps_full_state* state) {
    if (!state) {
        return nullptr;
    }
    return duplicateString(puzzlescript::exportSnapshot(*state->impl));
}

size_t ps_full_state_list_inputs(const ps_full_state*, ps_input* output, size_t capacity) {
    return puzzlescript::listInputs(output, capacity);
}

bool ps_benchmark_full_state_clone_hash(const ps_full_state* state, uint32_t iterations, uint32_t thread_count, ps_benchmark_result* out_result, ps_error** out_error) {
    if (out_error) {
        *out_error = nullptr;
    }
    if (!state || !out_result) {
        if (out_error) {
            *out_error = makeError(std::make_unique<Error>("ps_benchmark_full_state_clone_hash received null input"));
        }
        return false;
    }
    if (auto error = puzzlescript::benchmarkCloneHash(*state->impl, iterations, thread_count, *out_result)) {
        if (out_error) {
            *out_error = makeError(std::move(error));
        }
        return false;
    }
    return true;
}

void ps_runtime_counters_set_enabled(bool enabled) {
    puzzlescript::setRuntimeCountersEnabled(enabled);
}

void ps_runtime_counters_reset(void) {
    puzzlescript::resetRuntimeCounters();
}

void ps_runtime_counters_snapshot(ps_runtime_counters* out_counters) {
    if (out_counters) {
        *out_counters = puzzlescript::snapshotRuntimeCounters();
    }
}

int32_t ps_game_level_count(const ps_game* game) {
    return game && game->impl ? static_cast<int32_t>(game->impl->levels.size()) : 0;
}

int32_t ps_game_object_count(const ps_game* game) {
    return game && game->impl ? game->impl->objectCount : 0;
}

uint32_t ps_game_word_count(const ps_game* game) {
    return game && game->impl ? game->impl->wordCount : 0;
}

const char* ps_game_foreground_color(const ps_game* game) {
    return game && game->impl ? game->impl->foregroundColor.c_str() : "";
}

const char* ps_game_background_color(const ps_game* game) {
    return game && game->impl ? game->impl->backgroundColor.c_str() : "";
}

bool ps_game_has_metadata(const ps_game* game, const char* key_utf8) {
    if (!game || !game->impl || !key_utf8) {
        return false;
    }
    return game->impl->metadataMap.find(key_utf8) != game->impl->metadataMap.end();
}

const char* ps_game_metadata_value(const ps_game* game, const char* key_utf8) {
    if (!game || !game->impl || !key_utf8) {
        return "";
    }
    const auto it = game->impl->metadataMap.find(key_utf8);
    return it == game->impl->metadataMap.end() ? "" : it->second.c_str();
}

bool ps_game_sound_seed(const ps_game* game, const char* sound_name_utf8, int32_t* out_seed) {
    if (out_seed) {
        *out_seed = 0;
    }
    if (!game || !game->impl || !sound_name_utf8) {
        return false;
    }
    const auto it = game->impl->sfxEvents.find(sound_name_utf8);
    if (it == game->impl->sfxEvents.end()) {
        return false;
    }
    if (out_seed) {
        *out_seed = it->second;
    }
    return true;
}

bool ps_game_object_info(const ps_game* game, int32_t object_id, ps_object_info* out_info) {
    if (!game || !game->impl || !out_info || object_id < 0 || object_id >= game->impl->objectCount) {
        return false;
    }
    const auto& object = game->impl->objectsById[static_cast<size_t>(object_id)];
    out_info->name = object.name.c_str();
    out_info->id = object.id;
    out_info->layer = object.layer;
    out_info->color_count = object.colors.size();
    out_info->sprite_height = static_cast<int32_t>(object.sprite.size());
    out_info->sprite_width = object.sprite.empty() ? 0 : static_cast<int32_t>(object.sprite.front().size());
    return true;
}

const char* ps_game_object_color(const ps_game* game, int32_t object_id, size_t color_index) {
    if (!game || !game->impl || object_id < 0 || object_id >= game->impl->objectCount) {
        return "";
    }
    const auto& colors = game->impl->objectsById[static_cast<size_t>(object_id)].colors;
    return color_index < colors.size() ? colors[color_index].c_str() : "";
}

int32_t ps_game_object_sprite_value(const ps_game* game, int32_t object_id, int32_t x, int32_t y) {
    if (!game || !game->impl || object_id < 0 || object_id >= game->impl->objectCount || x < 0 || y < 0) {
        return -1;
    }
    const auto& sprite = game->impl->objectsById[static_cast<size_t>(object_id)].sprite;
    if (static_cast<size_t>(y) >= sprite.size() || static_cast<size_t>(x) >= sprite[static_cast<size_t>(y)].size()) {
        return -1;
    }
    return sprite[static_cast<size_t>(y)][static_cast<size_t>(x)];
}

const char* ps_error_message(const ps_error* error) {
    return error && error->impl ? error->impl->message.c_str() : "";
}

void ps_free_error(ps_error* error) {
    delete error;
}

void ps_free_game(ps_game* game) {
    delete game;
}

void ps_string_free(char* string_value) {
    delete[] string_value;
}
