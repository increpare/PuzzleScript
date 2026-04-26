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
typedef ps_session ps_full_state;
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

typedef struct ps_object_info {
    const char* name;
    int32_t id;
    int32_t layer;
    size_t color_count;
    int32_t sprite_width;
    int32_t sprite_height;
} ps_object_info;

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
typedef ps_session_mode ps_full_state_mode;

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
typedef ps_session_status_info ps_full_state_status_info;

typedef struct ps_step_result {
    bool changed;
    bool won;
    bool transitioned;
    bool restarted;
    size_t audio_event_count;
    const ps_audio_event* audio_events;
    size_t ui_audio_event_count;
    const ps_audio_event* ui_audio_events;
} ps_step_result;

typedef struct ps_compact_tick_oracle_info {
    bool attempted;
    bool handled;
    bool matched;
    bool state_checked;
    ps_step_result compact_result;
    ps_step_result interpreter_result;
} ps_compact_tick_oracle_info;
typedef ps_compact_tick_oracle_info ps_compact_turn_oracle_info;

typedef struct ps_benchmark_result {
    uint64_t iterations;
    uint32_t threads;
    double elapsed_seconds;
    double iterations_per_second;
    uint64_t hash_accumulator;
} ps_benchmark_result;

typedef struct ps_runtime_counters {
    uint64_t rules_visited;
    uint64_t rules_skipped_by_mask;
    uint64_t candidate_cells_tested;
    uint64_t pattern_tests;
    uint64_t pattern_matches;
    uint64_t replacements_attempted;
    uint64_t replacements_applied;
    uint64_t row_scans;
    uint64_t ellipsis_scans;
    uint64_t mask_rebuild_calls;
    uint64_t mask_rebuild_dirty_calls;
    uint64_t mask_rebuild_rows;
    uint64_t mask_rebuild_columns;
    uint64_t compiled_rule_group_attempts;
    uint64_t compiled_rule_group_hits;
    uint64_t compiled_rule_group_fallbacks;
    uint64_t compiled_tick_attempts;
    uint64_t compiled_tick_hits;
    uint64_t compiled_tick_fallbacks;
    uint64_t specialized_full_turn_attempts;
    uint64_t specialized_full_turn_hits;
    uint64_t specialized_full_turn_fallbacks;
    uint64_t specialized_rulegroup_attempts;
    uint64_t specialized_rulegroup_hits;
    uint64_t specialized_rulegroup_fallbacks;
} ps_runtime_counters;

bool ps_load_ir_json(const char* json_utf8, size_t json_size, ps_game** out_game, ps_error** out_error);
bool ps_compile_source(const char* source_utf8, size_t source_size, ps_compile_result** out_result);
const ps_game* ps_compile_result_game(const ps_compile_result* result);
const ps_error* ps_compile_result_error(const ps_compile_result* result);
void ps_free_compile_result(ps_compile_result* result);

bool ps_full_state_create(const ps_game* game, ps_full_state** out_state, ps_error** out_error);
bool ps_full_state_create_with_loaded_level_seed(
    const ps_game* game,
    const char* loaded_level_seed_utf8,
    ps_full_state** out_state,
    ps_error** out_error);
bool ps_full_state_clone(const ps_full_state* state, ps_full_state** out_state, ps_error** out_error);
void ps_full_state_destroy(ps_full_state* state);
void ps_full_state_set_unit_testing(ps_full_state* state, bool enabled);
bool ps_full_state_load_level(ps_full_state* state, int32_t level_index, ps_error** out_error);
ps_step_result ps_full_state_turn(ps_full_state* state, ps_input input);
bool ps_full_state_compact_tick_oracle_check(
    const ps_full_state* state,
    ps_input input,
    ps_compact_tick_oracle_info* out_info);
bool ps_full_state_compact_turn_oracle_check(
    const ps_full_state* state,
    ps_input input,
    ps_compact_turn_oracle_info* out_info);
bool ps_full_state_pending_again(const ps_full_state* state);
bool ps_full_state_undo(ps_full_state* state);
bool ps_full_state_restart(ps_full_state* state);
bool ps_full_state_advance_level(ps_full_state* state, ps_error** out_error);
void ps_full_state_status(const ps_full_state* state, ps_full_state_status_info* out_status);
const char* ps_full_state_message_text(const ps_full_state* state);
bool ps_full_state_cell_has_object(const ps_full_state* state, int32_t x, int32_t y, int32_t object_id);
bool ps_full_state_first_player_position(const ps_full_state* state, int32_t* out_x, int32_t* out_y);
uint64_t ps_full_state_hash64(const ps_full_state* state);
ps_hash128 ps_full_state_hash128(const ps_full_state* state);
char* ps_full_state_serialize_test_string(const ps_full_state* state);
char* ps_full_state_export_snapshot(const ps_full_state* state);
size_t ps_full_state_list_inputs(const ps_full_state* state, ps_input* output, size_t capacity);

/* Legacy session API names retained for compatibility. */
bool ps_session_create(const ps_game* game, ps_session** out_session, ps_error** out_error);
bool ps_session_create_with_loaded_level_seed(
    const ps_game* game,
    const char* loaded_level_seed_utf8,
    ps_session** out_session,
    ps_error** out_error);
bool ps_session_clone(const ps_session* session, ps_session** out_session, ps_error** out_error);
void ps_session_destroy(ps_session* session);
void ps_session_set_unit_testing(ps_session* session, bool enabled);
bool ps_session_load_level(ps_session* session, int32_t level_index, ps_error** out_error);
ps_step_result ps_session_step(ps_session* session, ps_input input);
ps_step_result ps_session_tick(ps_session* session);
bool ps_session_compact_tick_oracle_check(
    const ps_session* session,
    ps_input input,
    ps_compact_tick_oracle_info* out_info);
bool ps_session_pending_again(const ps_session* session);
bool ps_session_undo(ps_session* session);
bool ps_session_restart(ps_session* session);
bool ps_session_advance_level(ps_session* session, ps_error** out_error);
void ps_session_status(const ps_session* session, ps_session_status_info* out_status);
const char* ps_session_message_text(const ps_session* session);
bool ps_session_cell_has_object(const ps_session* session, int32_t x, int32_t y, int32_t object_id);
bool ps_session_first_player_position(const ps_session* session, int32_t* out_x, int32_t* out_y);
uint64_t ps_session_hash64(const ps_session* session);
ps_hash128 ps_session_hash128(const ps_session* session);
char* ps_session_serialize_test_string(const ps_session* session);
char* ps_session_export_snapshot(const ps_session* session);
size_t ps_session_list_inputs(const ps_session* session, ps_input* output, size_t capacity);
bool ps_benchmark_clone_hash(const ps_session* session, uint32_t iterations, uint32_t thread_count, ps_benchmark_result* out_result, ps_error** out_error);
void ps_runtime_counters_set_enabled(bool enabled);
void ps_runtime_counters_reset(void);
void ps_runtime_counters_snapshot(ps_runtime_counters* out_counters);

int32_t ps_game_level_count(const ps_game* game);
int32_t ps_game_object_count(const ps_game* game);
uint32_t ps_game_word_count(const ps_game* game);
const char* ps_game_foreground_color(const ps_game* game);
const char* ps_game_background_color(const ps_game* game);
bool ps_game_has_metadata(const ps_game* game, const char* key_utf8);
const char* ps_game_metadata_value(const ps_game* game, const char* key_utf8);
bool ps_game_sound_seed(const ps_game* game, const char* sound_name_utf8, int32_t* out_seed);
bool ps_game_object_info(const ps_game* game, int32_t object_id, ps_object_info* out_info);
const char* ps_game_object_color(const ps_game* game, int32_t object_id, size_t color_index);
int32_t ps_game_object_sprite_value(const ps_game* game, int32_t object_id, int32_t x, int32_t y);

const char* ps_error_message(const ps_error* error);
void ps_free_error(ps_error* error);
void ps_free_game(ps_game* game);
void ps_string_free(char* string_value);

#ifdef __cplusplus
}
#endif

#endif
