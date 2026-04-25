#include <cassert>
#include <cstring>
#include <iostream>
#include <string>

#include "puzzlescript/puzzlescript.h"

namespace {

constexpr const char* kSource = R"(title Native API Test
author Tests
text_color #ffffff
background_color #000000

========
OBJECTS
========

Background
black
00000
00000
00000
00000
00000

Player
white
.000.
.0.0.
.000.
..0..
..0..

=======
LEGEND
=======

. = Background
P = Player

======
SOUNDS
======

startgame 111111
Player MOVE 222222

================
COLLISIONLAYERS
================

Background
Player

======
RULES
======

[ Action Player ] -> [ Player ] message Hello native player

=======
LEVELS
=======

P.
)";

struct CompileHandle {
    ps_compile_result* result = nullptr;
    ~CompileHandle() { ps_free_compile_result(result); }
};

struct SessionHandle {
    ps_session* session = nullptr;
    ~SessionHandle() { ps_session_destroy(session); }
};

} // namespace

int main() {
    CompileHandle compiled;
    if (!ps_compile_source(kSource, std::strlen(kSource), &compiled.result)) {
        const ps_error* error = ps_compile_result_error(compiled.result);
        std::cerr << "compile failed: " << ps_error_message(error) << "\n";
        return 1;
    }

    const ps_game* game = ps_compile_result_game(compiled.result);
    assert(game != nullptr);
    assert(ps_game_level_count(game) == 1);
    assert(ps_game_object_count(game) >= 2);
    assert(std::string(ps_game_metadata_value(game, "title")) == "Native API Test");
    assert(std::string(ps_game_metadata_value(game, "author")) == "Tests");
    assert(std::string(ps_game_foreground_color(game)) == "#ffffff");
    assert(std::string(ps_game_background_color(game)) == "#000000");
    int32_t seed = 0;
    assert(ps_game_sound_seed(game, "startgame", &seed));
    assert(seed == 111111);

    ps_object_info playerInfo{};
    assert(ps_game_object_info(game, 1, &playerInfo));
    assert(playerInfo.sprite_width == 5);
    assert(playerInfo.sprite_height == 5);
    assert(playerInfo.color_count == 1);
    assert(std::string(ps_game_object_color(game, 1, 0)) == "white");
    assert(ps_game_object_sprite_value(game, 1, 1, 0) == 0);

    SessionHandle session;
    ps_error* error = nullptr;
    assert(ps_session_create(game, &session.session, &error));
    assert(ps_session_cell_has_object(session.session, 0, 0, 1));

    const ps_step_result result = ps_session_step(session.session, PS_INPUT_ACTION);
    assert(result.changed);
    ps_session_status_info status{};
    ps_session_status(session.session, &status);
    assert(status.mode == PS_SESSION_MODE_MESSAGE);
    assert(std::string(ps_session_message_text(session.session)) == "Hello native player");

    const ps_step_result closeResult = ps_session_step(session.session, PS_INPUT_ACTION);
    assert(closeResult.changed);
    ps_session_status(session.session, &status);
    assert(status.mode == PS_SESSION_MODE_LEVEL);
    assert(std::string(ps_session_message_text(session.session)).empty());

    const ps_step_result moveResult = ps_session_step(session.session, PS_INPUT_RIGHT);
    assert(moveResult.changed);
    assert(moveResult.audio_event_count == 1);
    assert(moveResult.audio_events[0].seed == 222222);

    ps_free_game(const_cast<ps_game*>(game));
    return 0;
}
