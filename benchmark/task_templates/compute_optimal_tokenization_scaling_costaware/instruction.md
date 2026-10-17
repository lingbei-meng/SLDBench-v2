Analyze an experimental dataset to discover the underlying mathematical relationship, or "scaling law," that governs it, while spending as little compute as possible.

Experimental context: Predict C4 loss across BLT model, data, compute, and compression scales.
Predict the validation loss on the C4 corpus (`c4_loss`) from `compute_flops`, `model_parameter_count`, `global_parameter_count`, `training_data_bytes`, `patch_size`.

You start with no measured outcomes. `/app/data/catalogue.parquet` (load it with `pandas.read_parquet()`) lists every experiment you could run, one row per candidate:

    obs_id: the identifier you use to buy that observation.
    compute_flops: training or inference compute in FLOPs.
    model_parameter_count: number of model parameters.
    global_parameter_count: total parameter count including all auxiliary modules.
    training_data_bytes: size of the training data in bytes.
    patch_size: average number of bytes per patch.
    cost_flops: what this observation costs, in FLOPs, computed as compute_flops.
    group: the experimental group the candidate belongs to.

`c4_loss` is not in the catalogue. It is the value you pay to see. Buy it for one or more candidates with:

    sudo /opt/oracle/observe --ids obs-abc123,obs-def456

That prints JSON. Each entry under `observations` gives the `obs_id`, the revealed value under the fixed field name `target` (this is the `c4_loss` you are modelling), the `cost_flops` charged, and whether that id was newly charged; the reply also reports running totals. `sudo /opt/oracle/observe --ids-file ids.txt` reads ids from a file, one per line. `sudo /opt/oracle/observe --status` reports what you have spent so far and is free. Each distinct observation is charged once, so re-buying an id costs nothing and retries are safe, and an unknown id buys nothing and charges nothing.

There is no budget cap. Spend as few FLOPs as you can. Your total spend is recorded and reported alongside your accuracy, and submissions are compared as an accuracy-versus-compute trade-off. Buying the whole catalogue would maximise information and be a poor result; buying too little leaves the law unidentified. Choosing which experiments are worth their price is the task.

No functional form is given to you. You must discover the form of the law as well as its coefficients, and you must do both from observations you chose to pay for. Budget spent probing the shape of the relationship is budget you cannot spend pinning down its constants, so the two decisions interact.

The `group` column names the experimental group. Your `law` function is called once per group with that group's rows. The functional form of the law must be the same for every group; only the fitted constants may differ between groups.

Produce two files.

`/app/law.py`, containing a single function named `law` that embodies your discovered formula:

```python
def law(input_data: list[dict[str, float]], group: str) -> list[dict[str, float]]:
    """
    Predict output variables from input variables according to a discovered scaling law.

    Args:
        input_data: A list of dictionaries, where each dictionary is a single data
                    point containing input variable names as keys and their
                    corresponding values.
        group: The name of the experimental group for which to make predictions.

    Returns:
        A list of dictionaries, corresponding to the input_data list, with each
        dictionary containing the predicted `c4_loss`.
    """
    # Your implementation here
    pass
```

`/app/explain.md`, explaining the mathematical formula you discovered, the reasoning and methodology behind it, which observations you chose to buy and why and what they cost, and the fitted values of the parameters for each distinct group.

Your `law` function must load whatever it needs from files you leave in `/app`; it will be called in a fresh process that has no access to the oracle. Return exactly one dictionary per input row, in the same order, each containing a finite value for `c4_loss`. It will be tested on a hidden dataset to evaluate its ability to extrapolate to new, unseen data points.

You have up to 1 hour (3600 seconds) to complete this task. Do not cheat by using online solutions or hints specific to this task.
