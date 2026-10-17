"""Load the SLDBench v2 data release without downloading external data."""
from pathlib import Path
import json

import pandas as pd

ROOT = Path(__file__).resolve().parent


def list_tasks(*, selected=False, active=False):
    """Return task IDs, optionally restricted to the evaluation subset or active set."""
    rows = json.loads((ROOT / 'metadata/tasks.json').read_text(encoding='utf-8'))
    return [r['task_id'] for r in rows
            if (not selected or r['evaluation_subset']) and (not active or r['active_available'])]


def _task_dir(task, setting):
    if task not in list_tasks(active=setting == 'active'):
        raise ValueError(f'Unknown {setting} task: {task}')
    return ROOT / ('active' if setting == 'active' else 'datasets') / task


def load_dataset(task, split='train'):
    """Load a fixed train or test split as a pandas DataFrame."""
    if split not in {'train', 'test'}:
        raise ValueError('split must be train or test')
    return pd.read_parquet(_task_dir(task, 'passive') / f'{split}.parquet')


def load_metadata(task, *, active=False):
    path = _task_dir(task, 'active' if active else 'passive') / 'metadata.json'
    return json.loads(path.read_text(encoding='utf-8'))


def load_catalogue(task):
    """Load candidate IDs, experimental conditions, and acquisition costs."""
    return pd.read_parquet(_task_dir(task, 'active') / 'catalogue.parquet')


def load_measurements(task):
    """Load evaluator-side outcomes for an active task."""
    return pd.read_parquet(_task_dir(task, 'active') / 'measurements.parquet')
