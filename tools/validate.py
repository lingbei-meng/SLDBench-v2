"""Validate all distributed tables, contracts, IDs, costs, and data hashes."""
from pathlib import Path
import hashlib
import json
import sys

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from sldbench import list_tasks, load_dataset, load_catalogue, load_measurements, load_metadata


def main():
    tasks = list_tasks()
    assert len(tasks) == len(set(tasks)) == 127
    assert len(list_tasks(selected=True)) == 30
    assert len(list_tasks(active=True)) == 60
    summary = json.loads((ROOT/'metadata/summary.json').read_text())
    counts = {'train': 0, 'test': 0}
    for task in tasks:
        meta = load_metadata(task)
        contract = meta['contract']
        required = list(dict.fromkeys(contract['features'] + [contract['target']]))
        for split in counts:
            frame = load_dataset(task, split)
            assert set(required).issubset(frame.columns), (task, split, required)
            assert len(frame) == meta['splits'][split], (task, split)
            assert np.isfinite(frame[required].to_numpy(dtype=float)).all(), (task, split)
            counts[split] += len(frame)
        if meta['active_available']:
            am = load_metadata(task, active=True)
            cat = load_catalogue(task)
            obs = load_measurements(task)
            assert cat.obs_id.is_unique and obs.obs_id.is_unique, task
            assert set(cat.obs_id) == set(obs.obs_id), task
            assert am['contract']['target'] not in cat.columns, task
            costs = cat.cost_flops.to_numpy(dtype=float)
            assert np.isfinite(costs).all() and (costs > 0).all(), task
            assert len(cat) == am['cost_model']['pool_rows'], task
            assert np.isclose(costs.sum(), am['cost_model']['pool_cost'], rtol=1e-12), task
    assert counts['train'] == summary['train_rows']
    assert counts['test'] == summary['test_rows']
    hashes = json.loads((ROOT/'metadata/data_sha256.json').read_text())
    for name, expected in hashes.items():
        assert hashlib.sha256((ROOT/name).read_bytes()).hexdigest() == expected, name
    print(json.dumps({'tasks':len(tasks), 'active_tasks':60, 'evaluation_tasks':30,
                      'train_records':counts['train'], 'test_records':counts['test'],
                      'prediction_records':sum(counts.values()),
                      'verified_data_files':len(hashes)}, indent=2))


if __name__ == '__main__':
    main()
