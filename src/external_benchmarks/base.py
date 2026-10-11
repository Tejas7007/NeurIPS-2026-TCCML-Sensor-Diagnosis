from dataclasses import dataclass, field
from typing import Dict, List, Optional
import numpy as np
import pandas as pd

FAULT_ORIGINS = ('native', 'synthetic')
FAULT_SCOPES = ('healthy', 'sensor', 'system', 'cyber', 'mixed', 'unknown')


HEALTHY = 'HEALTHY'
SENSOR_FAULT = 'SENSOR_FAULT'
SYSTEM_FAULT = 'SYSTEM_OR_PROCESS_FAULT'
MIXED_FAULT = 'MULTIPLE_OR_MIXED_FAULT'
UNSUPPORTED = 'UNSUPPORTED_OR_INSUFFICIENT_EVIDENCE'
DIAGNOSTIC_STATES = (HEALTHY, SENSOR_FAULT, SYSTEM_FAULT, MIXED_FAULT, UNSUPPORTED)


@dataclass
class ChannelRoles:

    trusted_context: List[str] = field(default_factory=list)
    monitored_sensors: List[str] = field(default_factory=list)
    control_actuator: List[str] = field(default_factory=list)
    potential_outputs: List[str] = field(default_factory=list)
    excluded: List[str] = field(default_factory=list)
    rationale: str = ''

    def validate(self, columns) -> None:
        cols = set(columns)
        for name in ('trusted_context', 'monitored_sensors', 'control_actuator',
                     'potential_outputs'):
            missing = [c for c in getattr(self, name) if c not in cols]
            if missing:
                raise ValueError(f'{name} references absent columns: {missing}')
        overlap = set(self.monitored_sensors) & set(self.control_actuator)
        if overlap:
            raise ValueError(f'a channel cannot be both monitored sensor and actuator: {overlap}')


@dataclass
class BenchmarkFrame:


    name: str
    data: pd.DataFrame
    roles: ChannelRoles
    fault_origin: pd.Series
    fault_scope: pd.Series
    faulty_channel: Optional[pd.Series] = None
    mechanism: Optional[pd.Series] = None
    manifestation: Optional[pd.Series] = None
    operating_condition: Optional[pd.Series] = None
    is_chronological: bool = False
    notes: str = ''

    def __post_init__(self):
        for col in ('unit_id', 't_index'):
            if col not in self.data.columns:
                raise ValueError(f'BenchmarkFrame.data must contain {col!r}')
        self.roles.validate(self.data.columns)
        bad = set(self.fault_origin.unique()) - set(FAULT_ORIGINS)
        if bad:
            raise ValueError(f'unknown fault_origin values: {bad}')
        bad = set(self.fault_scope.unique()) - set(FAULT_SCOPES)
        if bad:
            raise ValueError(f'unknown fault_scope values: {bad}')

    def summary(self) -> dict:
        return dict(
            name=self.name, rows=len(self.data), units=int(self.data.unit_id.nunique()),
            chronological=self.is_chronological,
            monitored=len(self.roles.monitored_sensors),
            context=len(self.roles.trusted_context),
            scope_counts={k: int(v) for k, v in self.fault_scope.value_counts().items()},
            origin_counts={k: int(v) for k, v in self.fault_origin.value_counts().items()},
        )


class BenchmarkAdapter:


    name = 'base'

    def available(self) -> bool:
        raise NotImplementedError

    def load(self) -> BenchmarkFrame:
        raise NotImplementedError

    def describe_unavailable(self) -> str:

        return f'{self.name}: raw data not present'

