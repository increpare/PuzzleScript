#include "runtime/core.hpp"
#include "runtime/compiled_rules.hpp"

#include <cstring>

 #include "compiler/lower_to_runtime.hpp"
 #include "compiler/parser.hpp"

using puzzlescript::CompileResult;
using puzzlescript::Error;
using puzzlescript::Game;
using puzzlescript::Session;

struct ps_game {
    std::shared_ptr<const Game> impl;
};

struct ps_session {
    std::unique_ptr<Session> impl;
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

bool ps_session_create(const ps_game* game, ps_session** out_session, ps_error** out_error) {
    if (out_error) {
        *out_error = nullptr;
    }
    if (!game || !out_session) {
        if (out_error) {
            *out_error = makeError(std::make_unique<Error>("ps_session_create received null input"));
        }
        return false;
    }
    auto* wrapper = new ps_session();
    wrapper->impl = puzzlescript::createSession(game->impl);
    *out_session = wrapper;
    return true;
}

bool ps_session_create_with_loaded_level_seed(
    const ps_game* game,
    const char* loaded_level_seed_utf8,
    ps_session** out_session,
    ps_error** out_error
) {
    if (out_error) {
        *out_error = nullptr;
    }
    if (!game || !out_session) {
        if (out_error) {
            *out_error = makeError(std::make_unique<Error>("ps_session_create_with_loaded_level_seed received null input"));
        }
        return false;
    }
    if (!loaded_level_seed_utf8) {
        return ps_session_create(game, out_session, out_error);
    }
    auto* wrapper = new ps_session();
    wrapper->impl = puzzlescript::createSessionWithLoadedLevelSeed(game->impl, loaded_level_seed_utf8);
    *out_session = wrapper;
    return true;
}

bool ps_session_clone(const ps_session* session, ps_session** out_session, ps_error** out_error) {
    if (out_error) {
        *out_error = nullptr;
    }
    if (!session || !out_session) {
        if (out_error) {
            *out_error = makeError(std::make_unique<Error>("ps_session_clone received null input"));
        }
        return false;
    }
    auto* wrapper = new ps_session();
    wrapper->impl = std::make_unique<Session>(*session->impl);
    *out_session = wrapper;
    return true;
}

void ps_session_destroy(ps_session* session) {
    delete session;
}

void ps_session_set_unit_testing(ps_session* session, bool enabled) {
    if (session == nullptr || !session->impl) {
        return;
    }
    session->impl->suppressRuleMessages = enabled;
}

bool ps_session_load_level(ps_session* session, int32_t level_index, ps_error** out_error) {
    if (out_error) {
        *out_error = nullptr;
    }
    if (!session) {
        if (out_error) {
            *out_error = makeError(std::make_unique<Error>("ps_session_load_level received null session"));
        }
        return false;
    }
    if (auto error = puzzlescript::loadLevel(*session->impl, level_index)) {
        if (out_error) {
            *out_error = makeError(std::move(error));
        }
        return false;
    }
    return true;
}

ps_step_result ps_session_step(ps_session* session, ps_input input) {
    if (!session) {
        return ps_step_result{};
    }
    return puzzlescript::step(*session->impl, input);
}

ps_step_result ps_session_tick(ps_session* session) {
    if (!session) {
        return ps_step_result{};
    }
    return puzzlescript::tick(*session->impl);
}

bool ps_session_pending_again(const ps_session* session) {
    return session && session->impl->pendingAgain;
}

bool ps_session_undo(ps_session* session) {
    return session ? puzzlescript::undo(*session->impl) : false;
}

bool ps_session_restart(ps_session* session) {
    return session ? puzzlescript::restart(*session->impl) : false;
}

bool ps_session_advance_level(ps_session* session, ps_error** out_error) {
    if (out_error) {
        *out_error = nullptr;
    }
    if (!session) {
        if (out_error) {
            *out_error = makeError(std::make_unique<Error>("ps_session_advance_level received null session"));
        }
        return false;
    }
    if (auto error = puzzlescript::advanceLevel(*session->impl)) {
        if (out_error) {
            *out_error = makeError(std::move(error));
        }
        return false;
    }
    return true;
}

void ps_session_status(const ps_session* session, ps_session_status_info* out_status) {
    if (!session || !out_status) {
        return;
    }
    out_status->mode = session->impl->preparedSession.titleScreen
        ? PS_SESSION_MODE_TITLE
        : (session->impl->preparedSession.textMode ? PS_SESSION_MODE_MESSAGE : PS_SESSION_MODE_LEVEL);
    out_status->current_level_index = session->impl->preparedSession.currentLevelIndex;
    out_status->has_current_level_target = session->impl->preparedSession.currentLevelTarget.has_value();
    out_status->current_level_target = session->impl->preparedSession.currentLevelTarget.value_or(0);
    out_status->width = session->impl->liveLevel.width;
    out_status->height = session->impl->liveLevel.height;
    out_status->title_mode = session->impl->preparedSession.titleMode;
    out_status->title_selection = session->impl->preparedSession.titleSelection;
    out_status->can_undo = session->impl->canUndo;
    out_status->winning = session->impl->preparedSession.winning;
    out_status->title_screen = session->impl->preparedSession.titleScreen;
    out_status->text_mode = session->impl->preparedSession.textMode;
    out_status->title_selected = session->impl->preparedSession.titleSelected;
    out_status->message_selected = session->impl->preparedSession.messageSelected;
}

const char* ps_session_message_text(const ps_session* session) {
    if (!session || !session->impl) {
        return "";
    }
    const auto& prepared = session->impl->preparedSession;
    if (!prepared.messageText.empty()) {
        return prepared.messageText.c_str();
    }
    if (prepared.textMode && prepared.level.isMessage) {
        return prepared.level.message.c_str();
    }
    return "";
}

bool ps_session_cell_has_object(const ps_session* session, int32_t x, int32_t y, int32_t object_id) {
    if (!session || !session->impl || object_id < 0) {
        return false;
    }
    const Session& impl = *session->impl;
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

bool ps_session_first_player_position(const ps_session* session, int32_t* out_x, int32_t* out_y) {
    if (out_x) {
        *out_x = 0;
    }
    if (out_y) {
        *out_y = 0;
    }
    if (!session || !session->impl) {
        return false;
    }
    const Session& impl = *session->impl;
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

uint64_t ps_session_hash64(const ps_session* session) {
    return session ? puzzlescript::hashSession64(*session->impl) : 0;
}

ps_hash128 ps_session_hash128(const ps_session* session) {
    return session ? puzzlescript::hashSession128(*session->impl) : ps_hash128{};
}

char* ps_session_serialize_test_string(const ps_session* session) {
    if (!session) {
        return nullptr;
    }
    return duplicateString(puzzlescript::serializeTestString(*session->impl));
}

char* ps_session_export_snapshot(const ps_session* session) {
    if (!session) {
        return nullptr;
    }
    return duplicateString(puzzlescript::exportSnapshot(*session->impl));
}

size_t ps_session_list_inputs(const ps_session*, ps_input* output, size_t capacity) {
    return puzzlescript::listInputs(output, capacity);
}

bool ps_benchmark_clone_hash(const ps_session* session, uint32_t iterations, uint32_t thread_count, ps_benchmark_result* out_result, ps_error** out_error) {
    if (out_error) {
        *out_error = nullptr;
    }
    if (!session || !out_result) {
        if (out_error) {
            *out_error = makeError(std::make_unique<Error>("ps_benchmark_clone_hash received null input"));
        }
        return false;
    }
    if (auto error = puzzlescript::benchmarkCloneHash(*session->impl, iterations, thread_count, *out_result)) {
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
