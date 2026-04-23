#!/usr/bin/env bash
# Profile the C++ simulation replay workload.
#
# This is intentionally not a parity-test runner. It reuses the generated replay
# corpus as workload input, then runs `puzzlescript_cpp profile-simulations` so
# profiler output is dominated by C++ runtime IR loading/session/replay work
# rather than the JS harness. It does not measure PuzzleScript source
# parse/compile time; that should be a separate compiler profiling path.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$ROOT"

OUT="${PROFILE_STATS_OUT:-$ROOT/profile_stats.txt}"
PUZZLESCRIPT_CPP="${PUZZLESCRIPT_CPP:-$ROOT/build/native/puzzlescript_cpp}"
MANIFEST="${PROFILE_MANIFEST:-$ROOT/build/js-parity-data/fixtures.json}"
ART="$ROOT/build/native/profile_last"
PROFILE_MODE="${PROFILE_MODE:-auto}"
REPLAY_REPEATS="${PROFILE_REPLAY_REPEATS:-3}"
SAMPLE_SECONDS="${PROFILE_SAMPLE_SECONDS:-20}"
EXTRA_CLI_ARGS="${EXTRA_CLI_ARGS:-}"

mkdir -p "$ART"

if [[ ! -x "$PUZZLESCRIPT_CPP" ]]; then
  echo "Missing executable: $PUZZLESCRIPT_CPP" >&2
  echo "Try: make build" >&2
  exit 1
fi

if [[ ! -f "$MANIFEST" ]]; then
  echo "Missing replay corpus manifest: $MANIFEST" >&2
  echo "Try: make js-parity-data" >&2
  exit 1
fi

read -r -a EXTRA_ARGS <<<"$EXTRA_CLI_ARGS"
SIM_PROFILE_ARGS=(profile-simulations "$MANIFEST" --profile-timers --repeat "$REPLAY_REPEATS")
if [[ ${#EXTRA_ARGS[@]} -gt 0 ]]; then
  SIM_PROFILE_ARGS+=("${EXTRA_ARGS[@]}")
fi

append_section() {
  {
    echo
    echo "===== $1 ====="
    echo
    cat "$2"
  } | tee -a "$OUT"
}

time_command() {
  if /usr/bin/time -lp true >/dev/null 2>&1; then
    /usr/bin/time -lp "$@"
  elif /usr/bin/time -v true >/dev/null 2>&1; then
    /usr/bin/time -v "$@"
  else
    "$@"
  fi
}

{
  echo "===== PuzzleScript C++ simulation profiling ====="
  date -R
  uname -a
  sw_vers 2>/dev/null || true
  git rev-parse HEAD 2>/dev/null || true
  echo
  echo "===== Binary ====="
  file "$PUZZLESCRIPT_CPP"
  ls -la "$PUZZLESCRIPT_CPP"
  echo
  echo "===== Workload ====="
  echo "replay_corpus=$MANIFEST"
  echo "replay_repeats=$REPLAY_REPEATS"
  echo "profile_mode=$PROFILE_MODE"
  echo "command: $PUZZLESCRIPT_CPP ${SIM_PROFILE_ARGS[*]}"
  echo
} | tee "$OUT"

echo "----- Pass 1: wall clock + native timer breakdown -----" | tee -a "$OUT"
PASS1_STDOUT="$ART/pass1.stdout"
PASS1_STDERR="$ART/pass1.stderr"
set +e
time_command "$PUZZLESCRIPT_CPP" "${SIM_PROFILE_ARGS[@]}" >"$PASS1_STDOUT" 2>"$PASS1_STDERR"
PASS1_STATUS=$?
set -e

append_section "Pass 1 stdout" <(cat "$PASS1_STDOUT")
append_section "Pass 1 stderr (native_simulation_profile + resource usage)" \
  <({ grep -E '^native_simulation_profile' "$PASS1_STDERR" || true; echo '--- stderr tail ---'; tail -30 "$PASS1_STDERR"; })
echo "pass1_exit_status=$PASS1_STATUS" | tee -a "$OUT"

if [[ "$PROFILE_MODE" == "auto" ]]; then
  if [[ "$(uname -s)" == "Darwin" ]] && command -v sample >/dev/null 2>&1; then
    PROFILE_MODE="sample"
  elif command -v perf >/dev/null 2>&1; then
    PROFILE_MODE="perf"
  else
    PROFILE_MODE="none"
  fi
fi

if [[ "$PROFILE_MODE" == "sample" ]]; then
  echo "----- Pass 2: macOS sample(1) CPU stacks -----" | tee -a "$OUT"
  PASS2_STDOUT="$ART/pass2.stdout"
  PASS2_STDERR="$ART/pass2.stderr"
  SAMPLE_FILE="$ART/sample_puzzlescript_cpp.txt"
  rm -f "$PASS2_STDOUT" "$PASS2_STDERR" "$SAMPLE_FILE"

  set +m
  "$PUZZLESCRIPT_CPP" "${SIM_PROFILE_ARGS[@]}" >"$PASS2_STDOUT" 2>"$PASS2_STDERR" &
  CPID=$!

  sample "$CPID" "$SAMPLE_SECONDS" -mayDie -fullPaths -file "$SAMPLE_FILE" &
  SPID=$!

  set +e
  wait "$CPID"
  PASS2_STATUS=$?
  wait "$SPID"
  SAMPLE_STATUS=$?
  set -e

  append_section "Pass 2 stdout" <(cat "$PASS2_STDOUT")
  append_section "Pass 2 stderr (native_simulation_profile + tail)" \
    <({ grep -E '^native_simulation_profile' "$PASS2_STDERR" || true; echo '--- stderr tail ---'; tail -20 "$PASS2_STDERR"; })

  {
    echo "pass2_exit_status=$PASS2_STATUS"
    echo "sample_exit_status=$SAMPLE_STATUS"
    if [[ -f "$SAMPLE_FILE" ]]; then
      echo
      echo "===== Hot stacks: sample(1), sorted by top of stack ====="
      echo
      grep -A 140 "Sort by top of stack, same collapsed" "$SAMPLE_FILE" | head -110 || true
      echo
      echo "===== sample(1) call graph preview ====="
      echo
      head -140 "$SAMPLE_FILE" || true
    else
      echo
      echo "sample(1) did not produce a stack file. On recent macOS versions this can"
      echo "require elevated permissions; try the Instruments command printed below."
    fi
  } | tee -a "$OUT"
elif [[ "$PROFILE_MODE" == "perf" ]]; then
  echo "----- Pass 2: Linux perf record/report -----" | tee -a "$OUT"
  PASS2_STDOUT="$ART/pass2.stdout"
  PASS2_STDERR="$ART/pass2.stderr"
  PERF_DATA="$ART/perf.data"
  PERF_REPORT="$ART/perf_report.txt"
  rm -f "$PASS2_STDOUT" "$PASS2_STDERR" "$PERF_DATA" "$PERF_REPORT"

  set +e
  perf record -g -o "$PERF_DATA" -- "$PUZZLESCRIPT_CPP" "${SIM_PROFILE_ARGS[@]}" >"$PASS2_STDOUT" 2>"$PASS2_STDERR"
  PASS2_STATUS=$?
  perf report --stdio -i "$PERF_DATA" >"$PERF_REPORT" 2>/dev/null
  PERF_REPORT_STATUS=$?
  set -e

  append_section "Pass 2 stdout" <(cat "$PASS2_STDOUT")
  append_section "Pass 2 stderr (native_simulation_profile + perf output)" \
    <({ grep -E '^native_simulation_profile' "$PASS2_STDERR" || true; echo '--- stderr tail ---'; tail -30 "$PASS2_STDERR"; })

  {
    echo "pass2_exit_status=$PASS2_STATUS"
    echo "perf_report_exit_status=$PERF_REPORT_STATUS"
    echo
    echo "===== perf report preview ====="
    echo
    head -160 "$PERF_REPORT" || true
  } | tee -a "$OUT"
else
  echo "----- Pass 2 skipped: PROFILE_MODE=none or no profiler available -----" | tee -a "$OUT"
fi

{
  echo
  echo "===== Artifacts ====="
  echo "  $PASS1_STDOUT"
  echo "  $PASS1_STDERR"
  if [[ -n "${PASS2_STDOUT:-}" ]]; then echo "  $PASS2_STDOUT"; fi
  if [[ -n "${PASS2_STDERR:-}" ]]; then echo "  $PASS2_STDERR"; fi
  if [[ -f "$ART/sample_puzzlescript_cpp.txt" ]]; then echo "  $ART/sample_puzzlescript_cpp.txt"; fi
  if [[ -f "$ART/perf.data" ]]; then echo "  $ART/perf.data"; fi
  if [[ -f "$ART/perf_report.txt" ]]; then echo "  $ART/perf_report.txt"; fi
  echo
  echo "For Instruments UI on macOS:"
  echo "  xcrun xctrace record --template 'Time Profiler' --quiet --launch --output $ART/time_profiler.trace -- \\"
  echo "    \"$PUZZLESCRIPT_CPP\" ${SIM_PROFILE_ARGS[*]}"
} | tee -a "$OUT"

echo "Wrote $OUT (see also $ART/)" | tee -a "$OUT"
