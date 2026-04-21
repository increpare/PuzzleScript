#ifndef PUZZLESCRIPT_PUZZLESCRIPT_H
#define PUZZLESCRIPT_PUZZLESCRIPT_H

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct ps_game ps_game;
typedef struct ps_session ps_session;
typedef struct ps_compile_result ps_compile_result;
typedef struct ps_error ps_error;
typedef struct ps_level_view ps_level_view;

typedef struct ps_hash128 {
    uint64_t lo;
    uint64_t hi;
} ps_hash128;

typedef struct ps_audio_event {
    int32_t seed;
    const char* kind;
} ps_audio_event;

typedef enum ps_input {
    PS_INPUT_UP = 0,
    PS_INPUT_LEFT = 1,
    PS_INPUT_DOWN = 2,
    PS_INPUT_RIGHT = 3,
    PS_INPUT_ACTION = 4,
    PS_INPUT_TICK = 5
} ps_input;

typedef enum ps_session_mode {
    PS_SESSION_MODE_LEVEL = 0,
    PS_SESSION_MODE_TITLE = 1,
    PS_SESSION_MODE_MESSAGE = 2
} ps_session_mode;

typedef struct ps_session_status_info {
    ps_session_mode mode;
    int32_t current_level_index;
    bool has_current_level_target;
    int32_t current_level_target;
    int32_t width;
    int32_t height;
    int32_t title_mode;
    int32_t title_selection;
    bool can_undo;
    bool winning;
    bool title_screen;
    bool text_mode;
    bool title_selected;
    bool message_selected;
} ps_session_status_info;

typedef struct ps_step_result {
    bool changed;
    bool won;
    bool transitioned;
    size_t audio_event_count;
    const ps_audio_event* audio_events;
} ps_step_result;

typedef struct ps_benchmark_result {
    uint64_t iterations;
    uint32_t threads;
    double elapsed_seconds;
    double iterations_per_second;
    uint64_t hash_accumulator;
} ps_benchmark_result;

bool ps_load_ir_json(const char* json_utf8, size_t json_size, ps_game** out_game, ps_error** out_error);
bool ps_compile_source(const char* source_utf8, size_t source_size, ps_compile_result** out_result);
const ps_game* ps_compile_result_game(const ps_compile_result* result);
const ps_error* ps_compile_result_error(const ps_compile_result* result);
void ps_free_compile_result(ps_compile_result* result);

bool ps_session_create(const ps_game* game, ps_session** out_session, ps_error** out_error);
bool ps_session_clone(const ps_session* session, ps_session** out_session, ps_error** out_error);
void ps_session_destroy(ps_session* session);
bool ps_session_load_level(ps_session* session, int32_t level_index, ps_error** out_error);
ps_step_result ps_session_step(ps_session* session, ps_input input);
ps_step_result ps_session_tick(ps_session* session);
bool ps_session_undo(ps_session* session);
bool ps_session_restart(ps_session* session);
void ps_session_status(const ps_session* session, ps_session_status_info* out_status);
uint64_t ps_session_hash64(const ps_session* session);
ps_hash128 ps_session_hash128(const ps_session* session);
char* ps_session_serialize_test_string(const ps_session* session);
char* ps_session_export_snapshot(const ps_session* session);
size_t ps_session_list_inputs(const ps_session* session, ps_input* output, size_t capacity);
bool ps_benchmark_clone_hash(const ps_session* session, uint32_t iterations, uint32_t thread_count, ps_benchmark_result* out_result, ps_error** out_error);

const char* ps_error_message(const ps_error* error);
void ps_free_error(ps_error* error);
void ps_free_game(ps_game* game);
void ps_string_free(char* string_value);

#ifdef __cplusplus
}
#endif

#endif
