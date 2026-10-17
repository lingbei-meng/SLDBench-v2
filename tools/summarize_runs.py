"""Aggregate a completed run without modifying the published leaderboard."""
import argparse
from collections import defaultdict
import json
import math
from pathlib import Path
from statistics import mean, median


def aggregate(rows, configuration):
    repeats = configuration["repeats"]
    grouped = defaultdict(list)
    for row in rows:
        if row.get("error") or row.get("grader_error"):
            raise ValueError(f"Unsuccessful trial: {row['trial']}")
        score = row.get("r2")
        if not isinstance(score, (int, float)) or isinstance(score, bool) or not math.isfinite(score):
            raise ValueError(f"Non-finite or missing R²: {row['trial']}")
        grouped[(row["task"], row["setting"])].append(row)
    expected = {(t.removesuffix("_costaware"), "active" if t.endswith("_costaware") else "passive")
                for t in configuration["tasks"]}
    if set(grouped) != expected:
        raise ValueError("Run summary does not contain all configured tasks and settings")
    for key, values in grouped.items():
        if len(values) != repeats or len({r["trial"] for r in values}) != repeats:
            raise ValueError(f"Expected {repeats} distinct trials for {key}; found {len(values)}")
    summary = {"model": configuration["model"], "smoke": configuration.get("smoke", False),
               "repeats": repeats, "settings": {}}
    for key in ("agent", "agent_version", "protocol"):
        if key in configuration:
            summary[key] = configuration[key]
    for setting in ("passive", "active"):
        selected = {task: values for (task, s), values in grouped.items() if s == setting}
        if not selected:
            continue
        values = [row for batch in selected.values() for row in batch]
        result = {"tasks": len(selected), "runs": len(values),
                  "median_r2": median(r["r2"] for r in values)}
        if repeats == 3:
            best = {task: max(r["r2"] for r in batch) for task, batch in sorted(selected.items())}
            result.update({"median_best_at_3_r2": median(best.values()), "task_best_at_3_r2": best})
        if setting == "active":
            costs = [r.get("flops_fraction_of_pool") for r in values]
            if any(not isinstance(c, (int, float)) or not math.isfinite(c) or c < 0 or c > 1 + 1e-10 for c in costs):
                raise ValueError("Missing or invalid acquisition cost share")
            result.update({"mean_cost_share": mean(costs), "median_cost_share": median(costs)})
        summary["settings"][setting] = result
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run", type=Path, help="Run directory, e.g. runs/qwen_full")
    args = parser.parse_args()
    rows = json.loads((args.run / "summary.json").read_text(encoding="utf-8"))
    config = json.loads((args.run / "run-settings.json").read_text(encoding="utf-8"))
    result = aggregate(rows, config)
    text = json.dumps(result, indent=2, allow_nan=False) + "\n"
    (args.run / "aggregate.json").write_text(text, encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
