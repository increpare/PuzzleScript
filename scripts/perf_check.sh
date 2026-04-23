#!/usr/bin/env bash
# Run the native profile 5 times, take the median of each metric, compare
# against perf_baseline.json, and print a diff table. Exit 0 unconditionally
# (gatekeeping is done by humans reading the output, not this script).

set -eu

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

PUZZLESCRIPT_CPP="${PUZZLESCRIPT_CPP:-$ROOT/build/native/puzzlescript_cpp}"
MANIFEST="${PROFILE_MANIFEST:-$ROOT/build/js-parity-data/fixtures.json}"
BASELINE="$ROOT/perf_baseline.json"
RUNS="${PERF_RUNS:-5}"

if [[ ! -x "$PUZZLESCRIPT_CPP" ]]; then echo "missing $PUZZLESCRIPT_CPP — build native first" >&2; exit 2; fi
if [[ ! -f "$MANIFEST" ]]; then echo "missing $MANIFEST — run: make js-parity-data" >&2; exit 2; fi

TMP="$(mktemp -d)"
trap 'rm -rf "$TMP"' EXIT

for i in $(seq 1 "$RUNS"); do
  "$PUZZLESCRIPT_CPP" check-js-parity-data "$MANIFEST" --profile-timers >"$TMP/run$i.stdout" 2>"$TMP/run$i.stderr"
  awk -f "$ROOT/scripts/perf_extract.awk" "$TMP/run$i.stderr" > "$TMP/run$i.json"
done

python3 - "$TMP" "$RUNS" "$BASELINE" <<'PY'
import json, sys, os, statistics
tmp, runs, baseline_path = sys.argv[1], int(sys.argv[2]), sys.argv[3]
metrics = ["wall_ms","fast_replay_ms","game_load_ms","trace_json_parse_ms"]
samples = {m: [] for m in metrics}
for i in range(1, runs+1):
    with open(os.path.join(tmp, f"run{i}.json")) as f:
        d = json.load(f)
    for m in metrics: samples[m].append(int(d[m]))
median = {m: statistics.median(samples[m]) for m in metrics}
print(f"runs: {runs}")
for m in metrics:
    print(f"  {m}: samples={samples[m]} median={median[m]}")

if os.path.exists(baseline_path):
    with open(baseline_path) as f: baseline = json.load(f)
    print("\ndelta vs baseline:")
    for m in metrics:
        b = int(baseline[m]); c = median[m]
        pct = 100.0 * (c - b) / b if b else 0.0
        marker = "  OK" if c <= b * 1.02 else "REGR"
        print(f"  [{marker}] {m}: baseline={b} current={c} delta={c-b:+d} ({pct:+.1f}%)")
else:
    print(f"\n(no baseline at {baseline_path}; write current numbers there to establish one)")
PY
