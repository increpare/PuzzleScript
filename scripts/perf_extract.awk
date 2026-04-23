# Extracts profile metrics from a profile stderr file.
# Expects a line matching: native_simulation_profile test_cases=469 ... wall_ms=N ...
/^native_simulation_profile/ {
  for (i = 1; i <= NF; i++) {
    split($i, kv, "=")
    if (kv[1] == "wall_ms"             ) wall = kv[2]
    if (kv[1] == "replay_ms"           ) fast = kv[2]
    if (kv[1] == "game_load_ms"        ) load = kv[2]
    if (kv[1] == "trace_json_parse_ms" ) jsonp = kv[2]
  }
}
END {
  printf "{\"wall_ms\":%s,\"replay_ms\":%s,\"game_load_ms\":%s,\"trace_json_parse_ms\":%s}\n", wall, fast, load, jsonp
}
