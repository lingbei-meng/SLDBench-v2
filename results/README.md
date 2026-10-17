# Evaluation results

SLDBench v2 evaluates 30 tasks with three repeats in each setting. Results report held-out prediction quality and experiment acquisition cost separately.

| Result file | Contents |
|---|---|
| [leaderboard.json](leaderboard.json) | Aggregate quality and acquisition cost for all listed systems |
| [best_at_3.json](best_at_3.json) | Best@3 R² aggregates and the task-wise maximum across three repeats |
| [paper/summary.json](paper/summary.json) | The paper's main-table results and reference-law score |
| [claude_opus_5_5/summary.json](claude_opus_5_5/summary.json) | Claude Opus 5.5 aggregates and evaluation configuration |
| [claude_opus_5_5/README.md](claude_opus_5_5/README.md) | Claude Opus 5.5 per-task medians |
| [claude_opus_5_5/per_run.csv](claude_opus_5_5/per_run.csv) | Claude Opus 5.5 individual run results |

## Metrics

- **Passive / Active R²:** median held-out prediction score across runs in the corresponding setting.
- **Best@3 R²:** maximum held-out R² across the three repeats of each task, followed by the median across all 30 tasks; computed separately for Passive and Active.
- **Gap:** the Active median minus the Passive median.
- **Cost share:** the sum of prices of distinct acquired candidates divided by the complete catalogue's price.

## Claude Opus 5.5 configuration

The evaluation contains 180 runs: 30 tasks × 3 repeats × 2 settings. It uses Claude Code 2.1.280, model identifier `claude-opus-5-5[1m]`, a 1,000,000-token context window, high reasoning effort, a 3,600-second agent limit, and a 600-second verifier limit. Active acquisition proceeds without a fixed spending cap; quality and cost are reported separately.

## Run new evaluations

Use the [repository runner](../docs/RUNNING.md) to generate new results in `runs/`. The runner uses the 30 tasks listed in `metadata/evaluation_tasks.json`. Published result files in this directory are preserved independently of new runs.
