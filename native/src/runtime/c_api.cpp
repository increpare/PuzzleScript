#include "runtime/core.hpp"

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
