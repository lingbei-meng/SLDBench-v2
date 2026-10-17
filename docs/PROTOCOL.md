# Evaluation protocol

## Data and task selection

`datasets/` contains 127 fixed train/test datasets. `active/` supplies priced catalogues for 60 datasets. The executable evaluation uses the 30 task IDs in `metadata/evaluation_tasks.json`, with one Passive and one Active task definition per ID. Each pair uses identical held-out targets.

Runtime inputs are assembled from these canonical tables using `benchmark/tasks-manifest.json`. The manifest maps every generated file to its source and SHA-256 checksum. Candidate IDs, measurements, held-out data, task prompts, oracle code, and verifier code are checked before execution.

## Passive and Active observations

In Passive SLD, the agent receives the complete training table. In Active SLD, it receives a catalogue of candidate IDs, input features, groups, and FLOPs prices. Outcomes are revealed by a metered tool:

```bash
sudo /opt/oracle/observe --ids CANDIDATE_ID
sudo /opt/oracle/observe --status
```

Each distinct candidate is charged once. Repeated reads do not incur an additional charge. Status queries are free. Acquisition has no fixed spending cap in this evaluation; predictive quality and experiment cost are reported separately. The agent chooses what to purchase and when to stop.

The Active agent runs without root privileges. Unpurchased outcomes and the signed acquisition ledger are owned by the oracle. The verifier runs in a separate container with held-out data; candidate predictions execute under a restricted user. Runtime network access is restricted by Harbor to the configured model service.

## Submission and metrics

An agent submits `law.py` and `explain.md`. Each task defines the feature names and target returned by its prediction function:

```python
def law(input_data: list[dict[str, float]], group: str) -> list[dict[str, float]]:
    ...
```

The function returns one prediction dictionary for each input row, keyed by the task's target name. Task instructions specify the permitted model families and submission requirements.

- **R²:** `1 − sum((y − prediction)²) / sum((y − mean(y))²)` on held-out observations, as computed by the task verifier.
- **Cost share:** the sum of prices for distinct purchased candidates divided by the sum of all catalogue prices.
- **Median R²:** median held-out R² across runs in a setting.
- **Best@3 R²:** maximum R² across three repeats of each task and setting, followed by the median across tasks.
- **Gap:** Active median R² minus Passive median R².

Acquisition FLOPs describe the experiments represented by the observations. They are separate from model inference tokens and API charges. The runner records both acquisition metrics and inference usage.

## Published results and new runs

`results/` holds the published model results, including task-wise best@3 values. `runs/` holds new evaluations. The portable runner preserves task observations, prompts, acquisition rules, and scoring code while building its runtime from public dependencies. Its pinned harness configuration is listed in [RUNNING.md](RUNNING.md); historical model configurations remain associated with the published result files.
