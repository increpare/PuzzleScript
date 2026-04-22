#ifndef PUZZLESCRIPT_FRONTEND_H
#define PUZZLESCRIPT_FRONTEND_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct ps_frontend_result ps_frontend_result;

typedef enum ps_diagnostic_severity {
    PS_DIAG_ERROR = 0,
    PS_DIAG_WARNING = 1,
    PS_DIAG_INFO = 2,
    PS_DIAG_LOG = 3
} ps_diagnostic_severity;

typedef struct ps_diagnostic {
    ps_diagnostic_severity severity;
    int32_t code;
    int32_t line;
    const char* message;
} ps_diagnostic;

ps_frontend_result* ps_frontend_parse(const char* source_utf8, size_t source_size);
size_t ps_frontend_result_diagnostic_count(const ps_frontend_result* result);
const ps_diagnostic* ps_frontend_result_diagnostic(const ps_frontend_result* result, size_t index);
size_t ps_frontend_result_parser_state_json(
    const ps_frontend_result* result,
    char* out_buffer,
    size_t out_buffer_capacity
);
void ps_frontend_result_free(ps_frontend_result* result);

#ifdef __cplusplus
}
#endif

#endif
