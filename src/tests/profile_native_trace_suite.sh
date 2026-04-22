#!/usr/bin/env bash
# Profile the native trace sweep: timer breakdown + macOS `sample` CPU stacks + optional rusage.
# Writes repo-root profile_stats.txt and keeps large artifacts under build/native/profile_last/.
# pipefail off: pipelines using `head` may exit 141 (SIGPIPE); we still want the full script to finish.
set -eu

ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$ROOT"

OUT="${PROFILE_STATS_OUT:-$ROOT/profile_stats.txt}"
PS_CLI="${PS_CLI:-$ROOT/build/native/native/ps_cli}"
MANIFEST="${PROFILE_MANIFEST:-$ROOT/build/native/coverage-fixtures/fixtures.json}"
ART="$ROOT/build/native/profile_last"
EXTRA_CLI_ARGS="${EXTRA_CLI_ARGS:-}"

mkdir -p "$ART"

if [[ ! -x "$PS_CLI" ]]; then
  echo "Missing executable: $PS_CLI (build native first)" >&2
  exit 1
fi

if [[ ! -f "$MANIFEST" ]]; then
  echo "Missing manifest: $MANIFEST (run: make coverage-fixtures)" >&2
  exit 1
fi

{
  cat <<'DISCLAIMER'
I am an AI who implemented this really terribly/slowly, and who was inexplicably recalcitrant at the suggestion that C++ might be reasonably expected to be more performant than JS.  Here is a description of my crimes:

DISCLAIMER
  echo "===== PuzzleScript native trace profiling ====="
  date -R
  uname -a
  sw_vers 2>/dev/null || true
  git rev-parse HEAD 2>/dev/null || true
  echo
  echo "===== Binaries ====="
  file "$PS_CLI"
  ls -la "$PS_CLI"
  echo
  echo "===== sweep command ====="
  echo "$PS_CLI check-trace-sweep \"$MANIFEST\" --profile-timers $EXTRA_CLI_ARGS"
  echo
} | tee "$OUT"

append_section() {
  {
    echo
    echo "===== $1 ====="
    echo
    cat "$2"
  } | tee -a "$OUT"
}

echo "----- Pass 1: wall clock + getrusage (no sample overhead) -----" | tee -a "$OUT"
PASS1_STDOUT="$ART/pass1.stdout"
PASS1_STDERR="$ART/pass1.stderr"
set +e
/usr/bin/time -lp "$PS_CLI" check-trace-sweep "$MANIFEST" --profile-timers $EXTRA_CLI_ARGS \
  >"$PASS1_STDOUT" 2>"$PASS1_STDERR"
PASS1_STATUS=$?
set -e
append_section "Pass 1 stdout (tail: summary line)" <(tail -5 "$PASS1_STDOUT")
append_section "Pass 1 stderr (native_trace_suite_profile + tail of trace_case)" \
  <({ grep -E '^(native_trace_suite_profile|simulation_fixture_count=)' "$PASS1_STDERR" || true; echo '--- last 15 stderr lines ---'; tail -15 "$PASS1_STDERR"; })
{
  echo "pass1_exit_status=$PASS1_STATUS"
} | tee -a "$OUT"

echo "----- Pass 2: macOS sample(1) Time Profiler-style stacks (same workload) -----" | tee -a "$OUT"
PASS2_STDOUT="$ART/pass2.stdout"
PASS2_STDERR="$ART/pass2.stderr"
SAMPLE_FILE="$ART/sample_ps_cli.txt"

rm -f "$PASS2_STDOUT" "$PASS2_STDERR" "$SAMPLE_FILE"

set +m
"$PS_CLI" check-trace-sweep "$MANIFEST" --profile-timers $EXTRA_CLI_ARGS \
  >"$PASS2_STDOUT" 2>"$PASS2_STDERR" &
CPID=$!

sample "$CPID" 900 -mayDie -fullPaths -file "$SAMPLE_FILE" &
SPID=$!

set +e
wait "$CPID"
PASS2_STATUS=$?
set -e

set +e
wait "$SPID"
set -e

append_section "Pass 2 stdout (tail)" <(tail -5 "$PASS2_STDOUT")
append_section "Pass 2 stderr (profile line + tail)" \
  <({ grep -E '^(native_trace_suite_profile|simulation_fixture_count=)' "$PASS2_STDERR" || true; echo '--- last 10 stderr lines ---'; tail -10 "$PASS2_STDERR"; })

{
  echo "pass2_exit_status=$PASS2_STATUS"
  echo
  echo "===== Hot stacks: sample(1) 'Sort by top of stack' (>=5 hits) ====="
  echo
  grep -A 120 "Sort by top of stack, same collapsed" "$SAMPLE_FILE" | head -90 || true
  echo
  echo "===== sample(1) call graph (first 120 lines) ====="
  echo
  head -120 "$SAMPLE_FILE" || true
  echo
  echo "===== Artifacts (full logs; sample file is large) ====="
  echo "  $PASS1_STDOUT"
  echo "  $PASS1_STDERR"
  echo "  $PASS2_STDOUT"
  echo "  $PASS2_STDERR"
  echo "  $SAMPLE_FILE ($(wc -c < "$SAMPLE_FILE" | tr -d ' ') bytes, $(wc -l < "$SAMPLE_FILE" | tr -d ' ') lines)"
  echo
  echo "Open the sample file for full collapsed trees. For Instruments UI, run:"
  echo "  xcrun xctrace record --template 'Time Profiler' --quiet --launch --output $ART/time_profiler.trace -- \\"
  echo "    \"$PS_CLI\" check-trace-sweep \"$MANIFEST\" --profile-timers $EXTRA_CLI_ARGS"
} >>"$OUT"

echo "Wrote $OUT (see also $ART/ for full stderr/stdout and sample_ps_cli.txt)" | tee -a "$OUT"
