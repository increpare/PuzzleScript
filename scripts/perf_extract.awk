# Extracts profile metrics from a pass1.stderr file.
# Expects a line matching: native_trace_suite_profile simulation_fixtures=469 ... wall_ms=N ...
/^native_trace_suite_profile/ {
  for (i = 1; i <= NF; i++) {
    split($i, kv, "=")
    if (kv[1] == "wall_ms"             ) wall = kv[2]
    if (kv[1] == "fast_replay_ms"      ) fast = kv[2]
    if (kv[1] == "ir_miss_ms"          ) irmiss = kv[2]
    if (kv[1] == "trace_json_parse_ms" ) jsonp = kv[2]
  }
}
END {
  printf "{\"wall_ms\":%s,\"fast_replay_ms\":%s,\"ir_miss_ms\":%s,\"trace_json_parse_ms\":%s}\n", wall, fast, irmiss, jsonp
}
