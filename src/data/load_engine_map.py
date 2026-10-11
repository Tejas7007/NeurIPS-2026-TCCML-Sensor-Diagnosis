from pathlib import Path

import pandas as pd


CONTEXT = ["Engine Speed", "Torque"]
PRIMARY_MONITORED = ["T_IM", "P_IM", "EGR_Rate", "MF_IA", "NOx_EO"]
FUEL_MONITORED = "MF_FUEL"
ALL_USED = CONTEXT + PRIMARY_MONITORED + [FUEL_MONITORED]


def load_engine_map(path="data/engine_map.csv", verbose=True):
    source = Path(path)
    if source.suffix.lower() == ".csv":
        data = pd.read_csv(source)
        units = {
            "Engine Speed": "rpm",
            "Torque": "N.m",
            "T_IM": "degC",
            "P_IM": "bar",
            "EGR_Rate": "%",
            "MF_IA": "kg/h",
            "NOx_EO": "ppm",
            "MF_FUEL": "kg/h",
        }
    else:
        raw = pd.read_excel(source)
        units = raw.iloc[0].to_dict()
        data = raw.iloc[1:].reset_index(drop=True)
    missing = [column for column in ALL_USED if column not in data.columns]
    if missing:
        raise ValueError(f"Missing expected channels {missing}")
    for column in ALL_USED:
        data[column] = pd.to_numeric(data[column], errors="coerce")
    data = data.dropna(subset=ALL_USED).reset_index(drop=True)
    if verbose:
        print(f"Loaded {len(data)} operating points")
    return data, units


def assign_speed_blocks(data, n_speed_blocks=5):
    frame = data.copy()
    frame["speed_block"] = pd.qcut(
        frame["Engine Speed"], q=n_speed_blocks, labels=False, duplicates="drop"
    )
    frame["load_block"] = pd.qcut(
        frame["Torque"], q=3, labels=False, duplicates="drop"
    )
    frame["region_block"] = (
        frame["speed_block"].astype(str) + "_" + frame["load_block"].astype(str)
    )
    return frame
