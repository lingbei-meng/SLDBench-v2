# Development checks

Run the offline checks from the repository root:

```bash
python -m pip install -r requirements.txt -r requirements-baselines.txt
python tools/validate.py
python -m benchmark.prepare --data-only
python -m unittest discover -s tests -v
python tools/check_release.py
python run.py --all --repeats 3 --dry-run
```

These checks cover dataset contracts and checksums, catalogue integrity, task construction, held-out data isolation, framework routing, result aggregation, acquisition policies and baseline inputs. They also check Python syntax and local documentation links. The same checks run in GitHub Actions.

To test a framework through its model API, use the smoke-test commands in [RUNNING.md](RUNNING.md). To run the reference methods, follow [BASELINES.md](BASELINES.md).
