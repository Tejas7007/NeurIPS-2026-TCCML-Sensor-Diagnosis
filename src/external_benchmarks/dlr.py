from pathlib import Path
import json
import numpy as np
import pandas as pd

from src.external_benchmarks.base import BenchmarkAdapter, BenchmarkFrame, ChannelRoles

ROOT = Path(__file__).resolve().parents[2]
RAW = ROOT / 'data' / 'external' / 'dlr' / 'turbofan-sensor-fdi-bench'

SENSORS = ['NH', 'NL', 'WFE', 'PS0', 'P2', 'P023', 'P030', 'P044', 'P050', 'P134',
           'T2', 'T023', 'T030', 'T050']

CONTEXT = ['XM', 'ALT', 'DTISA', 'EPR']
HEALTH_PARAMS_PREFIX = ('DETA', 'CW')
SUITES = ('DS01', 'DS02', 'DS03', 'DS04')


def obs_col(sensor):
    return f'{sensor}_obs'


def truth_col(sensor):
    return f'{sensor}_truth'


class DLRTurbofanAdapter(BenchmarkAdapter):
    name = 'dlr_turbofan'

    def __init__(self, suite='DS03', split='test', raw_dir=RAW):
        self.suite = suite
        self.split = split
        self.raw_dir = Path(raw_dir)

    @property
    def suite_dir(self):
        return self.raw_dir / self.suite

    def available(self):
        return (self.suite_dir / 'parquet' / f'{self.split}.parquet').exists()

    def metadata(self):
        path = self.suite_dir / 'metadata' / 'suite_metadata.json'
        return json.loads(path.read_text()) if path.exists() else {}

    def _fault_flag_columns(self, df):

        return [c for c in df.columns if c.startswith('fault_') and c.endswith('_on')]

    def load(self):
        path = self.suite_dir / 'parquet' / f'{self.split}.parquet'
        if not path.exists():
            raise FileNotFoundError(f'{path} not found; see scripts/download_turbofan_data.py')
        df = pd.read_parquet(path)

        present = [s for s in SENSORS if obs_col(s) in df.columns]
        missing_truth = [s for s in present if truth_col(s) not in df.columns]
        if missing_truth:
            raise ValueError(f'observed channels without a paired truth column: {missing_truth}')

        df = df.rename(columns={'engine_id': 'unit_id', 'flight_cycle': 't_index'})
        df = df.sort_values(['unit_id', 't_index']).reset_index(drop=True)

        flags = self._fault_flag_columns(df)
        any_active = (df[flags].sum(axis=1) > 0) if flags else pd.Series(False, index=df.index)
        n_active = df[flags].sum(axis=1) if flags else pd.Series(0, index=df.index)

        scope = pd.Series(np.where(any_active.values, 'sensor', 'healthy'), index=df.index)
        origin = pd.Series('native', index=df.index)

        health_cols = [c for c in df.columns if c.startswith(HEALTH_PARAMS_PREFIX)]
        truth_cols = [truth_col(s) for s in present]
        label_cols = ['suite_id', 'fault_id'] + flags

        roles = ChannelRoles(
            trusted_context=[c for c in CONTEXT if c in df.columns],
            monitored_sensors=[obs_col(s) for s in present],
            control_actuator=[],
            excluded=truth_cols + health_cols + label_cols)

        keep = (['unit_id', 't_index'] + roles.trusted_context + roles.monitored_sensors
                + label_cols)
        data = df[[c for c in keep if c in df.columns]].copy()
        data['n_faults_active'] = n_active.values

        self._truth = df[truth_cols].copy()
        self._truth.columns = present
        self._present = present

        return BenchmarkFrame(
            name=f'dlr-{self.suite}-{self.split}', data=data, roles=roles,
            fault_origin=origin, fault_scope=scope,
            faulty_channel=None,
            mechanism=None,
            operating_condition=None,
            is_chronological=True)

    def oracle_values(self):


        if not hasattr(self, '_truth'):
            raise RuntimeError('call load() before oracle_values()')
        return self._truth.copy()


def parse_fault_targets(filename):


    stem = Path(filename).stem
    if '__engine_' not in stem:
        raise ValueError(f'unexpected filename form: {filename}')
    spec, engine = stem.split('__engine_')
    targets = {}
    current = None
    for token in spec.replace('__', '_').split('_'):
        if token in ('drift', 'step'):
            current = token
        elif token and current:
            targets.setdefault(current, []).append(token)
    if not targets:
        raise ValueError(f'no fault mechanism found in {filename}')
    return dict(engine_id=int(engine), targets=targets,
                faulted_sensors=sorted({s for v in targets.values() for s in v}))


def onset_cycle(frame, unit_id):

    d = frame.data
    sub = d[d.unit_id == unit_id]
    flags = [c for c in sub.columns if c.startswith('fault_') and c.endswith('_on')]
    if not flags:
        return None
    active = sub[sub[flags].sum(axis=1) > 0]
    return int(active.t_index.iloc[0]) if len(active) else None

