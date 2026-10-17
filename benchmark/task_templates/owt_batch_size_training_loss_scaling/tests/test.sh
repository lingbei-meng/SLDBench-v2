#!/bin/bash
set +e

reward_path="/logs/verifier/reward.txt"
metrics_path="/logs/verifier/metrics.json"
mkdir -p /logs/verifier

# Clear any reward that already exists before grading starts, so a planted file
# cannot survive as the trial's result.
rm -f "$reward_path" /logs/verifier/reward.json

failure_reason=""
if [ ! -f /app/law.py ]; then
  failure_reason="missing /app/law.py"
elif [ ! -f /app/explain.md ]; then
  failure_reason="missing /app/explain.md"
else
  cd "${TESTS_DIR:-/tests}" || failure_reason="cannot enter tests directory"
  if [ -z "$failure_reason" ]; then
    # guard.py destroys the held-out split and runs test_outputs.py, which is
    # unchanged, with the candidate law confined to a child process.
    python guard.py
    test_status=$?
    if [ "$test_status" -ne 0 ]; then
      failure_reason="guard.py exited with status $test_status"
    fi
  fi
fi

if [ -n "$failure_reason" ] || { [ ! -f "$reward_path" ] && [ ! -f /logs/verifier/reward.json ]; }; then
  printf '0.0\n' > "$reward_path"
  printf '{"oracle_reward": 0.0, "verifier_fallback": true, "reason": "%s"}\n' "$failure_reason" > "$metrics_path"
  printf 'Verifier fallback: %s; oracle_reward=0.0\n' "$failure_reason"
fi

exit 0
