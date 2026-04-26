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
sfx0 222222

================
COLLISIONLAYERS
================

Background
Player

======
RULES
======

[ > Player | Background ] -> [ Background | Player ] sfx0
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
    ps_full_state* state = nullptr;
    ~SessionHandle() { ps_full_state_destroy(state); }
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
    assert(ps_full_state_create(game, &session.state, &error));
    assert(ps_full_state_cell_has_object(session.state, 0, 0, 1));

    const ps_step_result result = ps_full_state_turn(session.state, PS_INPUT_ACTION);
    assert(result.changed);
    ps_full_state_status_info status{};
    ps_full_state_status(session.state, &status);
    assert(status.mode == PS_SESSION_MODE_MESSAGE);
    assert(std::string(ps_full_state_message_text(session.state)) == "Hello native player");

    const ps_step_result closeResult = ps_full_state_turn(session.state, PS_INPUT_ACTION);
    assert(closeResult.changed);
    ps_full_state_status(session.state, &status);
    assert(status.mode == PS_SESSION_MODE_LEVEL);
    assert(std::string(ps_full_state_message_text(session.state)).empty());

    const ps_step_result moveResult = ps_full_state_turn(session.state, PS_INPUT_RIGHT);
    assert(moveResult.changed);
    assert(moveResult.audio_event_count == 1);
    assert(moveResult.audio_events[0].seed == 222222);

    ps_free_game(const_cast<ps_game*>(game));
    return 0;
}
