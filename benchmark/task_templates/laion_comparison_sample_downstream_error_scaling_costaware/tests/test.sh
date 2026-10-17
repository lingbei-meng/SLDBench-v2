#!/bin/bash
# Always write a reward on every path so a crashed grader cannot silently
# yield "no result".
set +e

mkdir -p /logs/verifier
reward_json="/logs/verifier/reward.json"

# /logs/verifier is a world-writable bind mount and the candidate law runs in
# this container. Clear any reward that exists before grading starts, so a
# planted file cannot survive as the trial's result.
rm -f "$reward_json" /logs/verifier/reward.txt

cd "${TESTS_DIR:-/tests}" || {
  printf '{"reward": 0.0, "error": "cannot enter tests directory"}\n' > "$reward_json"
  printf '0.0\n' > /logs/verifier/reward.txt
  exit 0
}

python test_outputs.py
status=$?

if [ "$status" -ne 0 ] || [ ! -s "$reward_json" ]; then
  printf '{"reward": 0.0, "error": "verifier exited %d without writing a reward"}\n' "$status" > "$reward_json"
  printf '0.0\n' > /logs/verifier/reward.txt
fi

exit 0
