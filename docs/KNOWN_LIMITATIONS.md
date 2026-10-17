# Known limitations

This release retains two known issues in handling abnormal evaluation runs:

1. A Passive verifier failure can produce `verifier_fallback=true` and a zero reward. The summary and aggregation code may accept that fallback as a valid score of zero.
2. An Active acquisition ledger with invalid signatures can report `ledger_status=tampered` while omitting the rejected entries from the cost sum. The summary and aggregation code may accept the remaining cost, including zero, as a valid experiment cost.

Before using new runs in reported results, inspect the original verifier metrics and logs. Exclude failed verifiers and invalid or missing ledgers, investigate the cause, and rerun affected trials. A failed evaluation is not evidence of a model scoring zero, and an unverifiable ledger is not evidence of zero experiment cost.

These issues describe error handling; they do not establish that an agent can bypass the runtime's isolation or modify protected ledger files. They also do not establish that the published paper results are affected.

## Release verification scope

The supplied release passed its 22 offline tests, data validation, task materialization, and source/documentation checks in the local review. One full-pool reference task was executed successfully. This does not independently reproduce the paper's 30-task reference median of 0.455. The three CLI configurations were checked offline; real API calls and complete Docker-based agent evaluations were not independently executed in that review.
