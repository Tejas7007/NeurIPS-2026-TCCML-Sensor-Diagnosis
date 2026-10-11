from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.fuel_flow import GenericConsistencyMonitor, MONITORED, TARGET, split_inner
from src.data.load_engine_map import CONTEXT, load_engine_map
from src.evaluation.episode_protocol import DEV_GRID, TEST_GRID, MECHANISMS, derive_seed, inject, wilson_lower
from src.health.envelope_gate import EnvelopeGate

REGIME = "grouped_RPM_holdout"
METHODS = ("raw_reading", "always_reconstruct_fuel", "selective_reconstruction", "oracle_fault_mask")
SCHEDULES = ("uniform", "low_load", "high_load")
DEFAULT_PREVALENCE = (0.0, 0.001, 0.01, 0.05, 0.1)
EPS = 1e-10
CALIBRATION_COLUMNS = (
    "seed", "fold", "inner_fold", "physical_point_id", "scenario_id", "true_source",
    "mechanism", "margin", "truly_fuel_affected",
)


def _json_safe(value):
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(x) for x in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        return float(value) if np.isfinite(value) else None
    if isinstance(value, np.bool_):
        return bool(value)
    return value


def _write_json(path, data):
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(_json_safe(data), indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def _write_csv(path, frame):
    temporary = path.with_name(path.name + ".tmp")
    frame.to_csv(temporary, index=False)
    temporary.replace(path)


def _file_sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _data_path_label(path):
    resolved = Path(path).resolve()
    try:
        return resolved.relative_to(ROOT).as_posix()
    except ValueError:
        return resolved.name


def _config_sha256(config):
    serialized = json.dumps(_json_safe(config), sort_keys=True, allow_nan=False)
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()


def _save_fold_checkpoint(output, seed, fold, records, candidates, info, config_hash):
    points_path = output / f"points_seed{seed}_fold{fold}.csv"
    calibration_path = output / f"calibration_seed{seed}_fold{fold}.csv"
    status_path = output / f"checkpoint_seed{seed}_fold{fold}.json"
    _write_csv(points_path, records)
    _write_csv(calibration_path, candidates.reindex(columns=CALIBRATION_COLUMNS))

    _write_json(status_path, dict(
        completed=True, config_sha256=config_hash, seed=seed, fold=fold,
        points_file=points_path.name, points_sha256=_file_sha256(points_path),
        calibration_file=calibration_path.name, calibration_sha256=_file_sha256(calibration_path),
        evaluation_rows=len(records), calibration_rows=len(candidates), calibration_info=info,
    ))


def _load_fold_checkpoint(output, seed, fold, config_hash):
    status_path = output / f"checkpoint_seed{seed}_fold{fold}.json"
    if not status_path.exists():
        return None
    checkpoint = json.loads(status_path.read_text())
    if checkpoint.get("completed") is not True:
        return None
    if checkpoint.get("config_sha256") != config_hash or checkpoint.get("seed") != seed or checkpoint.get("fold") != fold:
        raise ValueError(f"Checkpoint configuration mismatch at {status_path}; use a new output directory")
    paths = {}
    for name in ("points", "calibration"):
        path = output / checkpoint[f"{name}_file"]
        if not path.is_file() or _file_sha256(path) != checkpoint[f"{name}_sha256"]:
            raise ValueError(f"Checkpoint {name} file is absent or changed at {path}; use a new output directory")
        paths[name] = path
    records = pd.read_csv(paths["points"], float_precision="round_trip")
    candidates = pd.read_csv(paths["calibration"], float_precision="round_trip")
    if len(records) != checkpoint["evaluation_rows"] or len(candidates) != checkpoint["calibration_rows"]:
        raise ValueError("Checkpoint row counts differ from the completion marker")
    if len(records) and ((records.seed != seed).any() or (records.fold != fold).any()):
        raise ValueError("Checkpoint contains records from a different seed/fold")
    info = checkpoint["calibration_info"]

    if info.get("threshold") is None:
        info["threshold"] = float("inf")
    for key in ("selected_precision", "selected_wilson_lower"):
        if info.get(key) is None:
            info[key] = np.nan
    return records, candidates, info


def schedule_weights(torque, schedule):

    t = np.asarray(torque, dtype=float)
    if not len(t) or not np.isfinite(t).all():
        raise ValueError("Schedules require a nonempty finite torque vector")
    q = (t - t.min()) / max(float(np.ptp(t)), EPS)
    if schedule == "uniform":
        w = np.ones(len(t))
    elif schedule == "low_load":
        w = 1.0 + 4.0 * (1.0 - q)
    elif schedule == "high_load":
        w = 1.0 + 4.0 * q
    else:
        raise ValueError(f"Unknown schedule {schedule!r}")
    return w / w.sum()


def accounting_metrics(reference, reading, weights, factor):

    ref = np.asarray(reference, dtype=float)
    val = np.asarray(reading, dtype=float)
    w = np.asarray(weights, dtype=float)
    if ref.shape != val.shape or ref.shape != w.shape or not np.isclose(w.sum(), 1.0):
        raise ValueError("Rates and one-hour schedule weights must have matching shapes")
    true_kg = float(factor * np.dot(w, ref))
    estimate_kg = float(factor * np.dot(w, val))
    bias = estimate_kg - true_kg
    point_abs_kg = float(factor * np.dot(w, np.abs(val - ref)))
    return {
        "true_co2_kg_per_engine_hour": true_kg,
        "estimated_co2_kg_per_engine_hour": estimate_kg,
        "signed_inventory_bias_kg": bias,
        "absolute_inventory_error_kg": abs(bias),
        "relative_signed_inventory_bias": bias / true_kg if abs(true_kg) > EPS else np.nan,
        "relative_absolute_inventory_error": abs(bias) / abs(true_kg) if abs(true_kg) > EPS else np.nan,
        "time_weighted_absolute_measurement_error_kg_co2": point_abs_kg,
    }


def frozen_fuel_action(detected, supported, predicted_source, margin, margin_threshold):

    return (
        np.asarray(detected, bool)
        & np.asarray(supported, bool)
        & (np.asarray(predicted_source) == TARGET)
        & (np.asarray(margin, float) >= float(margin_threshold))
    )


def _score(model, frame):

    z0 = model.z(frame)
    maximum = z0.max(axis=1)
    scores = np.zeros((len(frame), len(MONITORED)), dtype=float)
    fuel_reconstruction = None
    for j, sensor in enumerate(MONITORED):
        reconstructed = model.reconstruct(frame, sensor)
        if sensor == TARGET:
            fuel_reconstruction = reconstructed
        corrected = frame.copy()
        corrected[sensor] = reconstructed
        after = model.z(corrected).max(axis=1)
        scores[:, j] = (maximum - after) / (maximum + 1e-9)
    order = np.argsort(-scores, axis=1)
    predicted = np.asarray(MONITORED)[order[:, 0]]
    margin = scores[np.arange(len(frame)), order[:, 0]] - scores[np.arange(len(frame)), order[:, 1]]
    return maximum > model.threshold, predicted, margin, np.asarray(fuel_reconstruction)


def _grid(grid, quick):
    return {mechanism: values[:1] if quick else list(values) for mechanism, values in grid.items()}


def scenario_frames(frame, model, grid, seed, fold, stage, realizations):

    scenarios = [dict(
        scenario_id="healthy", true_source="none", mechanism="healthy", severity=0.0,
        direction=0, realization=0, frame=frame.copy(), manifestation=np.zeros(len(frame), bool),
        corruption_seed=0,
    )]
    for sensor in MONITORED:
        for mechanism in MECHANISMS:
            for severity in grid[mechanism]:
                directions = (1, -1) if mechanism in ("bias", "gain") else (1,)
                for direction in directions:
                    n_real = realizations if mechanism in ("noise", "partial_dropout") else 1
                    for real in range(n_real):
                        fseed = derive_seed(seed, fold, stage, sensor, mechanism, severity, direction, real)
                        corrupted, mask = inject(
                            frame, sensor, mechanism, severity, direction,
                            model.stds, model.sat_caps, fseed,
                        )
                        scenarios.append(dict(
                            scenario_id=f"{sensor}|{mechanism}|{severity:g}|{direction}|{real}",
                            true_source=sensor, mechanism=mechanism, severity=float(severity),
                            direction=direction, realization=real, frame=corrupted,
                            manifestation=np.asarray(mask, bool), corruption_seed=fseed,
                        ))
    return scenarios


def _score_scenarios(model, gate, scenarios):

    joined = pd.concat([scenario["frame"] for scenario in scenarios], ignore_index=True)
    detected, predicted, margins, fuel = _score(model, joined)
    supported = gate.inside(joined[CONTEXT].values)
    offset = 0
    for scenario in scenarios:
        end = offset + len(scenario["frame"])
        scenario["detected"] = detected[offset:end]
        scenario["predicted_source"] = predicted[offset:end]
        scenario["margin"] = margins[offset:end]
        scenario["reconstructed_fuel"] = fuel[offset:end]
        scenario["supported"] = supported[offset:end]
        offset = end
    return scenarios


def select_fuel_margin(train, seed, fold, *, trees=120, n_inner=4, quick=False,
                       target_precision=0.9, min_candidates=25, min_points=10):


    records = []
    splits = split_inner(train, REGIME, n_splits=n_inner, seed=seed)
    if len(splits) < 2:
        return float("inf"), dict(status="insufficient_inner_splits", threshold=np.inf), pd.DataFrame()
    for inner_fold, (train_idx, dev_idx) in enumerate(splits):
        fitting, development = train.iloc[train_idx].copy(), train.iloc[dev_idx].copy()
        if set(fitting.index) & set(development.index):
            raise AssertionError("Inner fitting and calibration physical points overlap")
        model = GenericConsistencyMonitor(seed=derive_seed(seed, fold, inner_fold, "calibration"),
                                          n_estimators=trees).fit(fitting, REGIME)
        gate = EnvelopeGate(k=3, quantile=0.95).fit(fitting[CONTEXT].values)
        scenarios = scenario_frames(development, model, _grid(DEV_GRID, quick), seed, fold,
                                    f"inner_{inner_fold}", realizations=1)
        _score_scenarios(model, gate, scenarios)
        for scenario in scenarios:

            candidates = scenario["detected"] & scenario["supported"] & (scenario["predicted_source"] == TARGET)
            raw = scenario["frame"][TARGET].to_numpy(float)
            true = development[TARGET].to_numpy(float)
            correct = (scenario["true_source"] == TARGET) & (np.abs(raw - true) > EPS)
            for j in np.flatnonzero(candidates):
                records.append(dict(
                    seed=seed, fold=fold, inner_fold=inner_fold,
                    physical_point_id=int(development.index[j]),
                    scenario_id=scenario["scenario_id"], true_source=scenario["true_source"],
                    mechanism=scenario["mechanism"], margin=float(scenario["margin"][j]),
                    truly_fuel_affected=int(correct[j]),
                ))
        print(f"  climate seed {seed} fold {fold}: calibration split {inner_fold + 1}/{len(splits)} complete", flush=True)
    calibration = pd.DataFrame(records)
    info = dict(
        status="target_unattainable", threshold=np.inf, n_candidates=len(calibration),
        n_distinct_points=int(calibration.physical_point_id.nunique()) if len(calibration) else 0,
        target_precision=target_precision, accepted_candidates=0, accepted_distinct_points=0,
        selected_precision=np.nan, selected_wilson_lower=np.nan,
    )
    if len(calibration) < min_candidates or info["n_distinct_points"] < min_points:
        info["status"] = "insufficient_candidates"
        return float("inf"), info, calibration
    best = None
    for threshold in np.unique(np.quantile(calibration.margin, np.linspace(0, 1, 81))):
        accepted = calibration[calibration.margin >= threshold]
        n = len(accepted)
        points = accepted.physical_point_id.nunique()
        if n < min_candidates or points < min_points:
            continue
        hits = int(accepted.truly_fuel_affected.sum())
        lower = wilson_lower(hits, n)
        if lower >= target_precision and (best is None or n > best["accepted_candidates"]):
            best = dict(status="selected", threshold=float(threshold), accepted_candidates=n,
                        accepted_distinct_points=int(points), selected_precision=hits / n,
                        selected_wilson_lower=lower)
    if best is not None:
        info.update(best)
    return float(info["threshold"]), info, calibration


def evaluate_scenarios(test, scenarios, seed, fold, threshold):
    records = []
    reference = test[TARGET].to_numpy(float)
    for scenario in scenarios:
        raw = scenario["frame"][TARGET].to_numpy(float)
        reconstructed = scenario["reconstructed_fuel"]
        action = frozen_fuel_action(scenario["detected"], scenario["supported"],
                                    scenario["predicted_source"], scenario["margin"], threshold)
        true_fuel_affected = (scenario["true_source"] == TARGET) & (np.abs(raw - reference) > EPS)
        selective = np.where(action, reconstructed, raw)

        oracle = np.where(true_fuel_affected, reconstructed, raw)
        for j in range(len(test)):
            records.append(dict(
                seed=seed, fold=fold, physical_point_id=int(test.index[j]),
                engine_speed_rpm=float(test[CONTEXT[0]].iloc[j]), torque_nm=float(test[CONTEXT[1]].iloc[j]),
                scenario_id=scenario["scenario_id"], true_source=scenario["true_source"],
                mechanism=scenario["mechanism"], severity=scenario["severity"],
                direction=scenario["direction"], realization=scenario["realization"],
                corruption_seed=scenario["corruption_seed"], manifestation=int(scenario["manifestation"][j]),
                fuel_affected=int(true_fuel_affected[j]), supported=int(scenario["supported"][j]),
                detected=int(scenario["detected"][j]), predicted_source=str(scenario["predicted_source"][j]),
                source_margin=float(scenario["margin"][j]), margin_threshold=threshold,
                acted_on_fuel=int(action[j]), true_fuel_kg_h=float(reference[j]),
                raw_fuel_kg_h=float(raw[j]), reconstructed_fuel_kg_h=float(reconstructed[j]),
                selective_fuel_kg_h=float(selective[j]), oracle_fuel_kg_h=float(oracle[j]),
                wrong_healthy_fuel_correction=int(action[j] and not true_fuel_affected[j]),
                false_fuel_source_selection=int(
                    scenario["manifestation"][j] and scenario["true_source"] not in (TARGET, "none")
                    and scenario["predicted_source"][j] == TARGET
                ),
                harmful_selective_correction=int(abs(selective[j] - reference[j]) > abs(raw[j] - reference[j]) + EPS),
            ))
    return pd.DataFrame(records)


def summarize_accounting(records, factor):
    rows = []


    scopes = [("all_covered_points", -1, records)]
    scopes += [("outer_fold", int(fold), group) for fold, group in records.groupby("fold")]
    columns = {
        "raw_reading": "raw_fuel_kg_h", "always_reconstruct_fuel": "reconstructed_fuel_kg_h",
        "selective_reconstruction": "selective_fuel_kg_h", "oracle_fault_mask": "oracle_fuel_kg_h",
    }
    for scope, fold, scope_records in scopes:
        for (seed, scenario_id), group in scope_records.groupby(["seed", "scenario_id"], sort=False):
            if group.physical_point_id.duplicated().any():
                raise AssertionError("Repeated physical points within one accounting scenario")
            first = group.iloc[0]
            reference = group.true_fuel_kg_h.to_numpy(float)
            raw = group.raw_fuel_kg_h.to_numpy(float)
            raw_abs = np.abs(raw - reference)
            unaffected = group.fuel_affected.to_numpy(int) == 0
            for schedule in SCHEDULES:
                weights = schedule_weights(group.torque_nm, schedule)
                before = float(np.dot(weights, raw_abs))
                for method, column in columns.items():
                    reading = group[column].to_numpy(float)
                    after_abs = np.abs(reading - reference)
                    if method == "raw_reading":
                        method_action = np.zeros(len(group), bool)
                    elif method == "always_reconstruct_fuel":
                        method_action = np.ones(len(group), bool)
                    elif method == "selective_reconstruction":
                        method_action = group.acted_on_fuel.to_numpy(int) == 1
                    else:
                        method_action = group.fuel_affected.to_numpy(int) == 1
                    metrics = accounting_metrics(reference, reading, weights, factor)
                    rows.append(dict(
                        scope=scope, fold=fold, seed=int(seed), scenario_id=scenario_id,
                        true_source=first.true_source, mechanism=first.mechanism, severity=first.severity,
                        direction=int(first.direction), realization=int(first.realization),
                        schedule=schedule, method=method, n_points=len(group), duration_hours=1.0,
                        n_fuel_affected=int(group.fuel_affected.sum()),
                        fuel_action_fraction=float(method_action.mean()),
                        time_weighted_fuel_action_fraction=float(np.dot(weights, method_action)),
                        wrong_healthy_fuel_correction_fraction=float((method_action & unaffected).mean()),
                        conditional_healthy_fuel_correction_fraction=float(method_action[unaffected].mean()) if unaffected.any() else np.nan,
                        harmful_point_fraction=float((after_abs > raw_abs + EPS).mean()),
                        time_weighted_harm_fraction=float(np.dot(weights, after_abs > raw_abs + EPS)),
                        added_absolute_error_kg_co2=float(factor * np.dot(weights, np.maximum(after_abs - raw_abs, 0))),
                        removed_absolute_error_kg_co2=float(factor * np.dot(weights, np.maximum(raw_abs - after_abs, 0))),
                        error_weighted_recovery=1.0 - float(np.dot(weights, after_abs)) / before if before > EPS else np.nan,
                        negative_fuel_estimate_fraction=float((reading < 0).mean()),
                        **metrics,
                    ))
    return pd.DataFrame(rows)


def summarize_prevalence(accounting, prevalences):


    rows = []
    metric_columns = [
        "signed_inventory_bias_kg", "absolute_inventory_error_kg",
        "time_weighted_absolute_measurement_error_kg_co2", "fuel_action_fraction",
        "wrong_healthy_fuel_correction_fraction", "harmful_point_fraction",
        "added_absolute_error_kg_co2", "removed_absolute_error_kg_co2",
    ]
    pooled = accounting[accounting.scope == "all_covered_points"]
    for (seed, schedule, method), group in pooled.groupby(["seed", "schedule", "method"]):
        healthy = group[group.mechanism == "healthy"]
        faults = group[group.mechanism != "healthy"]
        if len(healthy) != 1:
            raise AssertionError("Exactly one healthy accounting scenario expected")

        per_mechanism = faults.groupby(["true_source", "mechanism"])[metric_columns].mean()
        per_source = per_mechanism.groupby("true_source").mean()
        fault_groups = {
            "fuel_fault_hours": [TARGET],
            "other_sensor_fault_hours": [sensor for sensor in MONITORED if sensor != TARGET],
            "equal_six_sensor_fault_hours": list(MONITORED),
        }
        for fault_group, sources in fault_groups.items():
            conditional = per_source.loc[sources].mean()
            for prevalence in prevalences:
                row = dict(seed=int(seed), schedule=schedule, method=method, fault_prior=fault_group,
                           assumed_fault_hour_prevalence=float(prevalence),
                           assumed_fuel_source_probability_given_fault=1.0 if len(sources) == 1 else (1.0 / 6 if TARGET in sources else 0.0),
                           fault_source_weights="equal within named source group", mechanism_weights="equal across five mechanisms",
                           severity_direction_realization_weights="equal within each mechanism",
                           prevalence_unit="share of one-engine-hour scenarios")
                for metric in metric_columns:
                    healthy_value = float(healthy.iloc[0][metric])
                    fault_value = float(conditional[metric])
                    row[f"healthy_{metric}"] = healthy_value
                    row[f"fault_conditional_{metric}"] = fault_value
                    row[f"expected_{metric}"] = (1 - prevalence) * healthy_value + prevalence * fault_value
                rows.append(row)
    return pd.DataFrame(rows)


def summarize_harm(records):
    rows = []
    for (seed, source, mechanism, severity, direction), group in records.groupby(
        ["seed", "true_source", "mechanism", "severity", "direction"], dropna=False
    ):
        unaffected = group.fuel_affected == 0
        affected_other = (group.true_source != TARGET) & (group.true_source != "none") & (group.manifestation == 1)
        before = np.abs(group.raw_fuel_kg_h - group.true_fuel_kg_h)
        after = np.abs(group.selective_fuel_kg_h - group.true_fuel_kg_h)
        rows.append(dict(
            seed=int(seed), true_source=source, mechanism=mechanism, severity=float(severity), direction=int(direction),
            n_evaluation_points=len(group), n_unique_physical_points=int(group.physical_point_id.nunique()),
            n_detected=int(group.detected.sum()), n_supported=int(group.supported.sum()),
            n_fuel_actions=int(group.acted_on_fuel.sum()), n_fuel_unaffected=int(unaffected.sum()),
            n_wrong_healthy_fuel_corrections=int(group.wrong_healthy_fuel_correction.sum()),
            healthy_fuel_correction_rate=float(group.loc[unaffected, "acted_on_fuel"].mean()) if unaffected.any() else np.nan,
            n_other_sensor_affected=int(affected_other.sum()),
            false_fuel_source_selection_rate=float((group.loc[affected_other, "predicted_source"] == TARGET).mean()) if affected_other.any() else np.nan,
            erroneous_fuel_action_under_other_fault_rate=float(group.loc[affected_other, "acted_on_fuel"].mean()) if affected_other.any() else np.nan,
            n_harmful_selective_corrections=int(group.harmful_selective_correction.sum()),
            absolute_fuel_error_before_sum_kg_h=float(before.sum()), absolute_fuel_error_after_sum_kg_h=float(after.sum()),
            error_weighted_recovery=1.0 - float(after.sum()) / float(before.sum()) if before.sum() > EPS else np.nan,
        ))
    return pd.DataFrame(rows)


def _plot_prevalence(prevalence, output):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 3, figsize=(12, 3.6), sharey=True)
    titles = ("Fuel sensor faults", "Five non-fuel sensor faults", "Faults equally likely in six sensors")
    priors = ("fuel_fault_hours", "other_sensor_fault_hours", "equal_six_sensor_fault_hours")
    method_labels = {
        "raw_reading": "Recorded fuel flow",
        "always_reconstruct_fuel": "Replace every fuel reading",
        "selective_reconstruction": "Correct fuel when diagnosed",
        "oracle_fault_mask": "True fuel-fault locations",
    }
    for ax, prior, title in zip(axes, priors, titles):
        subset = prevalence[(prevalence.fault_prior == prior) & (prevalence.schedule == "uniform")]
        for method in METHODS:
            values = subset[subset.method == method].groupby("assumed_fault_hour_prevalence")["expected_absolute_inventory_error_kg"].mean()
            ax.plot(values.index, values.values, marker="o", label=method_labels[method])
        ax.set_title(title)
        ax.set_xlabel("Assumed share of fault hours")
        ax.grid(alpha=0.2)
    axes[0].set_ylabel("Expected absolute CO₂ error\n(kg per one-hour scenario)")
    handles, labels = axes[-1].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, -0.02), ncol=2)
    fig.tight_layout()
    for extension in ("png", "pdf"):
        fig.savefig(output / f"climate_prevalence.{extension}", bbox_inches="tight", dpi=180)
    plt.close(fig)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-path", default=str(ROOT / "data" / "engine_map.csv"))
    parser.add_argument("--output", type=Path, default=ROOT / "outputs" / "climate")
    parser.add_argument("--resume", action="store_true", help="Reload matching completed folds; data, code, and run configuration must match")
    parser.add_argument("--quick", action="store_true", help="Smoke test: 16 trees, reduced grids, two inner splits, one outer fold by default")
    parser.add_argument("--seeds", nargs="+", type=int, default=[42])
    parser.add_argument("--fold-limit", type=int, default=None)
    parser.add_argument("--realizations", type=int, default=3)
    parser.add_argument("--co2-factor", type=float, default=3.1863, help="kg fossil combustion CO2 per kg diesel; configurable default")
    parser.add_argument("--prevalences", nargs="+", type=float, default=list(DEFAULT_PREVALENCE))
    parser.add_argument("--target-source-precision", type=float, default=0.90)
    parser.add_argument("--minimum-calibration-candidates", type=int, default=25)
    parser.add_argument("--minimum-calibration-points", type=int, default=10)
    args = parser.parse_args(argv)
    if args.co2_factor <= 0 or args.realizations < 1 or any(p < 0 or p > 1 for p in args.prevalences):
        parser.error("Factor/realizations must be positive and prevalences must lie in [0, 1]")
    if not 0 < args.target_source_precision < 1 or args.minimum_calibration_candidates < 1 or args.minimum_calibration_points < 1:
        parser.error("Calibration precision must lie in (0,1); minimum counts must be positive")
    if len(set(args.seeds)) != len(args.seeds):
        parser.error("--seeds must contain distinct values")
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    frame, units = load_engine_map(args.data_path)
    if units.get(TARGET) != "kg/h":
        raise ValueError("Fuel rate units must be kg/h; convert the source data explicitly before running")
    frame["rpm_level"] = frame["Engine Speed"].round(0)
    levels = frame.rpm_level.nunique()
    if levels < 7:
        raise ValueError(f"Seven outer speed splits require at least seven speed levels; found {levels}")
    splits = list(GroupKFold(7).split(frame, groups=frame.rpm_level))
    fold_limit = args.fold_limit if args.fold_limit is not None else (1 if args.quick else 7)
    if not 1 <= fold_limit <= 7:
        parser.error("--fold-limit must be between 1 and 7")
    splits = splits[:fold_limit]
    trees = 16 if args.quick else 120
    inner = 2 if args.quick else 4
    realizations = 1 if args.quick else args.realizations
    config = dict(
        experiment="fuel_measurement_and_fossil_combustion_co2_accounting", smoke_test=args.quick,
        checkpoint_format_version=1,
        implementation_sha256={str(path.relative_to(ROOT)): _file_sha256(path) for path in (
            Path(__file__).resolve(), ROOT / "experiments" / "fuel_flow.py",
            ROOT / "src" / "faults" / "injectors.py", ROOT / "src" / "health" / "envelope_gate.py",
            ROOT / "src" / "evaluation" / "episode_protocol.py", ROOT / "src" / "data" / "load_engine_map.py",
        )},
        full_seven_fold_coverage=(fold_limit == 7), data_path=_data_path_label(args.data_path),
        data_sha256=hashlib.sha256(Path(args.data_path).read_bytes()).hexdigest(),
        seeds=args.seeds, outer_regime=REGIME, outer_folds_run=fold_limit, inner_splits=inner,
        ensemble_trees=trees, monitored_sensors=MONITORED, context_variables=CONTEXT,
        excluded_predictors=["BSFC", "q_MI"], test_grid=_grid(TEST_GRID, args.quick),
        development_grid=_grid(DEV_GRID, args.quick), stochastic_realizations=realizations,
        co2_factor_kg_per_kg_diesel=args.co2_factor,
        fault_hour_prevalences=args.prevalences,
        calibration_target_precision=args.target_source_precision,
        minimum_calibration_candidates=args.minimum_calibration_candidates,
        minimum_calibration_points=args.minimum_calibration_points,
    )
    config_path = output / "config.json"
    config_hash = _config_sha256(config)
    if config_path.exists():
        previous_config = json.loads(config_path.read_text())
        if _config_sha256(previous_config) != config_hash:
            parser.error("Output directory contains a different data/code/run configuration; choose a new --output directory")
    elif list(output.glob("checkpoint_seed*_fold*.json")) or list(output.glob("points_seed*_fold*.csv")):
        parser.error("Output directory has fold records without a configuration manifest; choose a new --output directory")
    _write_json(config_path, config)
    print(f"Climate {'SMOKE' if args.quick else 'full'} run: {len(args.seeds)} seed(s), {fold_limit} speed fold(s), {trees} trees, CPU", flush=True)
    start = time.monotonic()
    all_records, calibration_records, calibration_info = [], [], []
    completed = 0
    computed = 0
    computed_seconds = 0.0
    total = len(args.seeds) * len(splits)
    for seed in args.seeds:
        for fold, (train_idx, test_idx) in enumerate(splits):
            fold_start = time.monotonic()
            if args.resume:
                cached = _load_fold_checkpoint(output, seed, fold, config_hash)
                if cached is not None:
                    records, candidates, info = cached
                    all_records.append(records)
                    calibration_info.append(info)
                    if len(candidates):
                        calibration_records.append(candidates)
                    completed += 1
                    print(f"Climate seed {seed} fold {fold}: resumed verified checkpoint ({len(records)} evaluation rows)", flush=True)
                    continue
            train, test = frame.iloc[train_idx].copy(), frame.iloc[test_idx].copy()
            assert not (set(train.index) & set(test.index))
            assert not (set(train.rpm_level) & set(test.rpm_level))
            print(f"Climate seed {seed} fold {fold}: {len(train)} training, {len(test)} test points; training-only calibration", flush=True)
            threshold, info, candidates = select_fuel_margin(
                train, seed, fold, trees=trees, n_inner=inner, quick=args.quick,
                target_precision=args.target_source_precision, min_candidates=args.minimum_calibration_candidates,
                min_points=args.minimum_calibration_points,
            )
            info.update(seed=seed, fold=fold, train_points=len(train), test_points=len(test))
            calibration_info.append(info)
            if len(candidates):
                calibration_records.append(candidates)
            model = GenericConsistencyMonitor(seed=derive_seed(seed, fold, "outer_model"), n_estimators=trees).fit(train, REGIME)
            gate = EnvelopeGate(k=3, quantile=0.95).fit(train[CONTEXT].values)
            scenarios = scenario_frames(test, model, _grid(TEST_GRID, args.quick), seed, fold, "outer_test", realizations)
            _score_scenarios(model, gate, scenarios)
            records = evaluate_scenarios(test, scenarios, seed, fold, threshold)
            all_records.append(records)

            _save_fold_checkpoint(output, seed, fold, records, candidates, info, config_hash)
            _write_json(output / "calibration_status.json", calibration_info)
            completed += 1
            computed += 1
            computed_seconds += time.monotonic() - fold_start
            estimate = computed_seconds / computed * (total - completed)
            print(f"Climate fold finished in {time.monotonic() - fold_start:.1f}s; margin status={info['status']}; estimated remaining {estimate / 60:.1f}min", flush=True)
    records = pd.concat(all_records, ignore_index=True)
    accounting = summarize_accounting(records, args.co2_factor)
    prevalence = summarize_prevalence(accounting, args.prevalences)
    harm = summarize_harm(records)
    _write_csv(output / "per_point.csv", records)
    _write_csv(output / "accounting.csv", accounting)
    _write_csv(output / "prevalence.csv", prevalence)
    _write_csv(output / "harm.csv", harm)
    _write_csv(output / "calibration_summary.csv", pd.DataFrame(calibration_info))
    all_candidates = pd.concat(calibration_records, ignore_index=True) if calibration_records else pd.DataFrame(columns=CALIBRATION_COLUMNS)
    _write_csv(output / "calibration_candidates.csv", all_candidates)
    _plot_prevalence(prevalence, output)
    _write_json(output / "run_status.json", dict(completed=True, elapsed_seconds=time.monotonic() - start,
                                                evaluation_records=len(records), accounting_rows=len(accounting),
                                                smoke_test=args.quick, full_seven_fold_coverage=(fold_limit == 7),
                                                computed_folds=computed, resumed_folds=completed - computed))
    print(f"Climate completed in {(time.monotonic() - start) / 60:.1f}min; saved {output}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

