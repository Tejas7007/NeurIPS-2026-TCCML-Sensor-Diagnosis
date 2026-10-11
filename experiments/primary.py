from __future__ import annotations

import argparse
from dataclasses import asdict, replace
import hashlib
import itertools
import json
import os
from pathlib import Path
import time
import warnings

import joblib
import numpy as np
import pandas as pd
from sklearn.covariance import LedoitWolf
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold, KFold
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from src.data.load_engine_map import load_engine_map, assign_speed_blocks, CONTEXT
from src.evaluation import episode_protocol as ep
from src.evaluation.crossfit import oof_map
from src.health.shi import MONITORED

VERSION = "camera-ready-primary-1"
VARIANTS = ("candidate", "adjusted", "covariance", "pca", "autoencoder")
REGIMES = {"heldout": "grouped_RPM_holdout", "within": "within_map"}
SEVERITY_GRID = {
    "bias": [0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0],
    "gain": [0.02, 0.05, 0.08, 0.12, 0.16, 0.20, 0.30],
    "noise": [], "saturation": [60, 70, 80, 84, 88, 92, 96],
    "partial_dropout": [],
}


def _path_label(path):
    resolved = Path(path).resolve()
    root = Path(__file__).resolve().parents[1]
    try:
        return resolved.relative_to(root).as_posix()
    except ValueError:
        return resolved.name


def log(message):
    print(f"[primary {time.strftime('%H:%M:%S')}] {message}", flush=True)


def atomic_joblib(value, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    joblib.dump(value, temporary, compress=3)
    os.replace(temporary, path)


def atomic_csv(frame, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    frame.to_csv(temporary, index=False)
    os.replace(temporary, path)


class SourceModels:


    def __init__(self, seed, variants, quick=False):
        self.seed = int(seed)
        self.variants = tuple(variants)
        self.quick = bool(quick)

    def fit(self, train):
        self.base = ep.FoldModels(seed=self.seed, shi_normalization="ensemble").fit(train)
        caps = set(ep.DEV_GRID["saturation"]) | set(ep.TEST_GRID["saturation"]) | set(SEVERITY_GRID["saturation"])
        self.base.sat_caps = {
            c: {p: float(np.percentile(train[c].values, p)) for p in caps}
            for c in MONITORED
        }
        if "covariance" in self.variants:
            signed_oof = oof_map(train, self.base.shi._fit_parity,
                                 self.base.shi._residuals, n_folds=5, seed=self.seed)
            self.residual_scale = np.maximum(np.std(signed_oof, axis=0), 1e-9)
            self.cov = LedoitWolf().fit(signed_oof / self.residual_scale)
        if any(v in self.variants for v in ("pca", "autoencoder")):
            self.value_scaler = StandardScaler().fit(train[CONTEXT + MONITORED].values)
            healthy_values = self.value_scaler.transform(train[CONTEXT + MONITORED].values)
            if "pca" in self.variants:
                self.pca = PCA(n_components=3, svd_solver="full").fit(healthy_values)
            if "autoencoder" in self.variants:
                self.autoencoder = MLPRegressor(
                    hidden_layer_sizes=(8, 2, 8), activation="tanh",
                    solver="lbfgs", alpha=0.01,
                    max_iter=150 if self.quick else 1000,
                    random_state=self.seed,
                )
                with warnings.catch_warnings(record=True) as caught:
                    warnings.simplefilter("always", ConvergenceWarning)
                    self.autoencoder.fit(healthy_values, healthy_values)
                self.autoencoder_converged = not any(
                    isinstance(w.message, ConvergenceWarning) for w in caught)
        return self

    def score(self, frame):
        scored = self.base.score(frame)
        scores = {"candidate": (scored["smat"], list(scored["cands"]))}
        if "adjusted" in self.variants:
            scores["adjusted"] = (self.base.shi.channel_z(frame), list(MONITORED))
        if "covariance" in self.variants:
            signed = self.base.shi._residuals(self.base.shi.parity, frame)
            standardized = signed / self.residual_scale
            centered = standardized - self.cov.location_
            conditional = np.abs(centered @ self.cov.precision_) / np.sqrt(
                np.maximum(np.diag(self.cov.precision_), 1e-9))
            scores["covariance"] = (conditional, list(MONITORED))
        if any(v in self.variants for v in ("pca", "autoencoder")):
            values = self.value_scaler.transform(frame[CONTEXT + MONITORED].values)
            if "pca" in self.variants:
                reconstructed = self.pca.inverse_transform(self.pca.transform(values))
                scores["pca"] = (np.abs(values - reconstructed)[:, len(CONTEXT):], list(MONITORED))
            if "autoencoder" in self.variants:
                reconstructed = self.autoencoder.predict(values)
                scores["autoencoder"] = (np.abs(values - reconstructed)[:, len(CONTEXT):], list(MONITORED))
        scored["source_scores"] = scores
        return scored


def score_scenarios(models, frame, grid, cfg, regime, fold, stage):
    scenarios = [dict(kind="healthy", scen_id="healthy", mech="healthy", true_src=None,
                      severity=np.nan, direction=0, realization_id=0,
                      scored=models.score(frame), mask=np.zeros(len(frame), bool))]
    for sensor in MONITORED:
        for mechanism in ep.MECHANISMS:
            for severity in grid.get(mechanism, []):
                directions = [1, -1] if mechanism in ("bias", "gain") else [1]
                for direction in directions:
                    scenario_id = f"{sensor}|{mechanism}|{severity}|{direction}"
                    n_real = cfg.n_realizations if mechanism in ep.STOCHASTIC else 1
                    for realization in range(n_real):
                        seed = ep.derive_seed(cfg.seed, regime, fold, stage, scenario_id, realization)
                        corrupted, mask = ep.inject(frame, sensor, mechanism, severity, direction,
                                                     models.base.stds, models.base.sat_caps, seed)

                        mask = mask & ~np.isclose(corrupted[sensor].values, frame[sensor].values,
                                                  rtol=0, atol=1e-12)
                        scenarios.append(dict(kind="fault", scen_id=scenario_id, mech=mechanism,
                                              true_src=sensor, severity=severity, direction=direction,
                                              realization_id=realization, scored=models.score(corrupted),
                                              mask=mask, fseed=seed))
    return scenarios


def score_stress_scenarios(models, frame, cfg, regime, fold):

    out = []
    for sensor in MONITORED:
        for mechanism in ("stuck_at", "load_dependent_bias"):
            for direction in (-1, 1):
                changed = frame.copy()
                if mechanism == "stuck_at":

                    value = models.training_medians[sensor] + direction * models.base.stds[sensor]
                    changed[sensor] = value
                else:

                    z_load = (frame["Torque"].values - models.training_context_mean[1]) / models.training_context_std[1]
                    changed[sensor] += direction * models.base.stds[sensor] * z_load
                mask = ~np.isclose(frame[sensor].values, changed[sensor].values)
                scenario = f"{sensor}|{mechanism}|1|{direction}"
                out.append(dict(kind="fault", scen_id=scenario, mech=mechanism, true_src=sensor,
                                severity=1.0, direction=direction, realization_id=0,
                                scored=models.score(changed), mask=mask,
                                true_sources=sensor,
                                fseed=ep.derive_seed(cfg.seed, regime, fold, "stress", scenario)))
    for first, second in itertools.combinations(MONITORED, 2):
        changed = frame.copy()
        changed[first] += 1.5 * models.base.stds[first]
        changed[second] -= 1.5 * models.base.stds[second]
        scenario = f"{first}+{second}|two_sensor_bias|1.5|0"
        out.append(dict(kind="fault", scen_id=scenario, mech="two_sensor_bias", true_src=None,
                        true_sources=f"{first};{second}", severity=1.5, direction=0,
                        realization_id=0, scored=models.score(changed), mask=np.ones(len(frame), bool),
                        fseed=ep.derive_seed(cfg.seed, regime, fold, "stress", scenario)))
    return out


def episode_records(models, scenarios, N, mode, cfg, regime, fold, stage, row_ids, variant):
    selected = []
    for scenario in scenarios:
        sc = dict(scenario)
        score = dict(sc["scored"])
        score["smat"], score["cands"] = score["source_scores"][variant]
        sc["scored"] = score
        selected.append(sc)
    records = ep.episodes_from_scenarios(models.base, selected, N, mode, cfg,
                                         regime=regime, outer_fold=fold,
                                         stage=stage, row_ids=row_ids)
    source_map = {s["scen_id"]: s.get("true_sources", s.get("true_src") or "none") for s in selected}
    for r in records:
        r["true_sources"] = source_map[r["scen_id"]]
    return records


class ConstantCalibrator:
    def __init__(self, p):
        self.p = float(p)

    def predict_proba(self, X):
        return np.tile([1.0 - self.p, self.p], (len(X), 1))


def align_probabilities(classifier, features):
    p = classifier.predict_proba(features)
    out = np.zeros((len(features), len(ep.MECHANISMS)))
    for i, name in enumerate(ep.MECHANISMS):
        if name in classifier.classes_:
            out[:, i] = p[:, list(classifier.classes_).index(name)]
    return out


def fit_reporting(records, k_alarm, cfg, trees, threads, population="fault_and_healthy"):


    fault_records = [r for r in records if r["mech"] in ep.MECHANISMS and r["true_src"] != "none"]
    if len(fault_records) < 20:
        raise ValueError("Need at least 20 development fault batches for reporting calibration")
    if population == "fault_only":
        batch_records = fault_records
    elif population == "fault_and_healthy":
        batch_records = [r for r in records if r["mech"] in ep.MECHANISMS + ["healthy"]]
    else:
        raise ValueError(f"Unknown reporting-calibration population {population!r}")
    X = np.asarray([ep.mech_features(r) for r in batch_records], float)
    y = np.asarray([r["mech"] for r in batch_records])
    nominal_fault = np.isin(y, ep.MECHANISMS)
    manifested = np.asarray([r.get("manifested", True) for r in batch_records], bool)
    is_fault = nominal_fault & manifested
    groups = np.asarray([r["inner_fold"] for r in batch_records])
    unique = np.unique(groups)
    if len(unique) < 3:
        raise ValueError("Reporting calibration requires at least three physical development partitions")

    def rf(seed):
        return RandomForestClassifier(n_estimators=trees, class_weight="balanced",
                                      random_state=seed, n_jobs=threads)

    def make_cal(features, labels):
        if len(np.unique(labels)) == 1:
            return ConstantCalibrator(np.mean(labels))
        return make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000)).fit(features, labels)

    def correctness(indices, probabilities):
        return np.asarray([int(batch_records[i].get("manifested", True) and
                               ep.complete_correct(batch_records[i], probabilities[j], k_alarm))
                           for j, i in enumerate(indices)])

    def features(indices, probabilities):
        return np.asarray([ep.correctness_features(batch_records[i], probabilities[j])
                           for j, i in enumerate(indices)], float)

    all_oof_prob = np.zeros((len(batch_records), len(ep.MECHANISMS)))
    doubly_oof_score = np.zeros(len(batch_records))
    doubly_oof_correct = np.zeros(len(batch_records), int)
    for held_group in unique:
        train_indices = np.flatnonzero(groups != held_group)
        test_indices = np.flatnonzero(groups == held_group)
        train_prob = np.zeros((len(train_indices), len(ep.MECHANISMS)))
        for nested_group in np.unique(groups[train_indices]):
            train_nested = train_indices[groups[train_indices] != nested_group]
            validation_nested = train_indices[groups[train_indices] == nested_group]
            train_faults = train_nested[is_fault[train_nested]]
            model = rf(ep.derive_seed(cfg.seed, "mechanism", held_group, nested_group)).fit(X[train_faults], y[train_faults])
            where = np.flatnonzero(groups[train_indices] == nested_group)
            train_prob[where] = align_probabilities(model, X[validation_nested])
        calibration = make_cal(features(train_indices, train_prob), correctness(train_indices, train_prob))
        train_faults = train_indices[is_fault[train_indices]]
        model = rf(ep.derive_seed(cfg.seed, "mechanism", held_group)).fit(X[train_faults], y[train_faults])
        validation_prob = align_probabilities(model, X[test_indices])
        all_oof_prob[test_indices] = validation_prob
        doubly_oof_score[test_indices] = calibration.predict_proba(features(test_indices, validation_prob))[:, 1]
        doubly_oof_correct[test_indices] = correctness(test_indices, validation_prob)
    indices = np.arange(len(batch_records))
    final_calibration = make_cal(features(indices, all_oof_prob), correctness(indices, all_oof_prob))
    final_mechanism = rf(ep.derive_seed(cfg.seed, "mechanism", "final")).fit(X[is_fault], y[is_fault])
    threshold = ep.tune_confidence(doubly_oof_score, doubly_oof_correct, cfg)
    return dict(mechanism=final_mechanism, reporting=final_calibration,
                threshold=threshold, n_development=len(batch_records),
                n_development_faults=int(is_fault.sum()),
                n_development_healthy=int((y == "healthy").sum()),
                n_development_nonmanifested=int((nominal_fault & ~manifested).sum()),
                reporting_population=population,
                n_physical_development_partitions=len(unique),
                group_policy="inner_physical_partition", calibration_policy="double_nested",
                development_scores=doubly_oof_score, development_correct=doubly_oof_correct)


def predict_records(records, fitted, k_alarm, seed, variant, task):

    features = np.asarray([ep.mech_features(r) for r in records], float)
    probabilities = align_probabilities(fitted["mechanism"], features)
    reporting_features = np.asarray([ep.correctness_features(r, probabilities[i])
                                     for i, r in enumerate(records)], float)
    scores = fitted["reporting"].predict_proba(reporting_features)[:, 1]
    threshold = fitted["threshold"]["thr"]
    out = []
    for i, r in enumerate(records):
        detected = r["n_suspicious"] >= k_alarm
        supported = r["support_frac"] > 0
        raw_mechanism = ep.MECHANISMS[int(probabilities[i].argmax())]
        manifested = bool(r.get("manifested", True))
        source_ok = detected and manifested and r["pred_src"] == r["true_src"] and r["mech"] != "healthy"
        type_ok = detected and manifested and raw_mechanism == r["mech"]
        known_fault = r["mech"] in ep.MECHANISMS
        accepted = bool(scores[i] >= threshold)
        complete = int(detected and supported and source_ok and type_ok) if known_fault else np.nan
        row = {k: v for k, v in r.items() if not k.startswith("f_") and not k.startswith("_")}
        row.update(seed=seed, variant=variant, task=task, mechanism=r["mech"],
                   detected=int(detected), supported=int(supported),
                   source_correct=int(source_ok), type_correct=int(type_ok),
                   predicted_mechanism=raw_mechanism,
                   complete_correct=complete, correctness_score=float(scores[i]),
                   frozen_threshold=float(threshold), accepted_frozen=int(accepted),
                   k_alarm=int(k_alarm), known_fault=int(known_fault),
                   calibration_status=fitted["threshold"]["status"],
                   group_policy=fitted["group_policy"],
                   reporting_population=fitted.get("reporting_population", "fault_only"))
        for j, mechanism in enumerate(ep.MECHANISMS):
            row[f"prob_{mechanism}"] = float(probabilities[i, j])
        out.append(row)
    return pd.DataFrame(out)


def safe_ratio(a, b):
    return float(a / b) if b else np.nan


def counts_and_metrics(frame):
    accepted = frame.accepted_frozen.astype(bool)
    detected = frame.detected.astype(bool)
    complete = frame.complete_correct.fillna(0).astype(bool)
    n = len(frame)
    n_detected = int(detected.sum())
    n_reported = int(accepted.sum())
    source_count = int(frame.source_correct.sum())
    return dict(n_batches=n, n_detected=n_detected, n_source_correct=source_count,
                n_type_correct=int(frame.type_correct.sum()), n_complete_correct=int(complete.sum()),
                n_reported=n_reported, n_correct_reports=int((accepted & complete).sum()),
                n_manifested=int(frame.manifested.astype(bool).sum()),
                detection=safe_ratio(n_detected, n),
                source_given_detected=safe_ratio(source_count, n_detected),
                type_given_detected=safe_ratio(int(frame.type_correct.sum()), n_detected),
                complete_rate=safe_ratio(int(complete.sum()), n),
                reporting_rate=safe_ratio(n_reported, n),
                complete_correct_given_report=safe_ratio(int((accepted & complete).sum()), n_reported))


def summaries(records):

    group_columns = ["task", "variant", "regime", "N", "mode"]
    rows = []
    for keys, group in records.groupby(group_columns, dropna=False):
        identifier = dict(zip(group_columns, keys))
        for view in ("all_injections", "manifested_batches"):
            known = group[group.known_fault == 1]
            if view == "manifested_batches":
                known = known[known.manifested.astype(bool)]
            if not len(known):
                continue
            mechanism_metrics = []
            for mechanism, part in known.groupby("mechanism"):
                metric = counts_and_metrics(part)
                mechanism_metrics.append(metric)
                rows.append(dict(identifier, view=view, summary=mechanism, **metric))
            rows.append(dict(identifier, view=view, summary="pooled", **counts_and_metrics(known)))
            metric_names = ["detection", "source_given_detected", "type_given_detected",
                            "complete_rate", "reporting_rate", "complete_correct_given_report"]
            macro = {m: float(np.mean([r[m] for r in mechanism_metrics])) for m in metric_names}
            rows.append(dict(identifier, view=view, summary="macro_average", n_batches=len(known),
                             n_mechanisms=len(mechanism_metrics),
                             n_mechanisms_with_reports=sum(r["n_reported"] > 0 for r in mechanism_metrics),
                             **macro))

            per_mechanism = pd.DataFrame(mechanism_metrics)
            rates = per_mechanism.div(per_mechanism.n_batches, axis=0)
            equal = dict(detection=float(rates.n_detected.mean()),
                         reporting_rate=float(rates.n_reported.mean()),
                         complete_rate=float(rates.n_complete_correct.mean()),
                         source_given_detected=safe_ratio(rates.n_source_correct.sum(), rates.n_detected.sum()),
                         type_given_detected=safe_ratio(rates.n_type_correct.sum(), rates.n_detected.sum()),
                         complete_correct_given_report=safe_ratio(rates.n_correct_reports.sum(), rates.n_reported.sum()))
            rows.append(dict(identifier, view=view, summary="equal_mechanism_prior", n_batches=len(known),
                             n_mechanisms=len(mechanism_metrics), **equal))
    return pd.DataFrame(rows)


def healthy_summary(records):
    healthy = records[records.mechanism == "healthy"]
    columns = ["task", "variant", "regime", "N", "mode", "seed", "outer_fold"]
    rows = []
    for keys, frame in healthy.groupby(columns):
        rows.append(dict(zip(columns, keys), n_healthy_batches=len(frame),
                         n_healthy_alarms=int(frame.detected.sum()),
                         n_healthy_diagnosis_reports=int(frame.accepted_frozen.sum()),
                         healthy_alarm_rate=float(frame.detected.mean()),
                         healthy_diagnosis_reporting_rate=float(frame.accepted_frozen.mean()),
                         healthy_report_without_alarm_rate=float(
                             ((frame.accepted_frozen == 1) & (frame.detected == 0)).mean())))
    return pd.DataFrame(rows)


def severity_summary(records, per_sensor=False):
    rows = []
    columns = ["variant", "regime", "N", "mode", "mechanism", "severity", "direction"]
    if per_sensor:
        columns.append("true_src")
    subset = records[(records.task == "severity") & (records.known_fault == 1)]
    for keys, frame in subset.groupby(columns):
        for view in ("all_injections", "manifested_batches"):
            use = frame if view == "all_injections" else frame[frame.manifested.astype(bool)]
            if len(use):
                rows.append(dict(zip(columns, keys), view=view, **counts_and_metrics(use),
                                 affected_point_fraction=float(use.manifested_fraction.mean()),
                                 mean_support_fraction=float(use.support_frac.mean())))
    return pd.DataFrame(rows)


def risk_coverage(records):
    rows = []
    columns = ["task", "variant", "regime", "N", "mode"]
    for keys, all_cases in records.groupby(columns):
        frame = all_cases[all_cases.known_fault == 1]
        if not len(frame):
            continue
        healthy = all_cases[all_cases.mechanism == "healthy"]
        scores = frame.correctness_score.to_numpy(float)
        correct = frame.complete_correct.to_numpy(float)
        for threshold in np.unique(np.quantile(scores, np.linspace(0, 1, 101))):
            selected = scores >= threshold
            n = int(selected.sum())
            rows.append(dict(zip(columns, keys), threshold=float(threshold),
                             n_batches=len(frame), n_accepted=n,
                             n_correct=int(correct[selected].sum()),
                             coverage=float(selected.mean()),
                             precision=float(correct[selected].mean()) if n else np.nan,
                             risk=float(1 - correct[selected].mean()) if n else np.nan,
                             healthy_reporting_rate=float((healthy.correctness_score >= threshold).mean())
                             if len(healthy) else np.nan,
                             frozen_operating_point=False))
        selected = frame.accepted_frozen.astype(bool).to_numpy()
        n = int(selected.sum())
        rows.append(dict(zip(columns, keys), threshold=np.nan, n_batches=len(frame),
                         n_accepted=n, n_correct=int(correct[selected].sum()),
                         coverage=float(selected.mean()),
                         precision=float(correct[selected].mean()) if n else np.nan,
                         risk=float(1 - correct[selected].mean()) if n else np.nan,
                         healthy_reporting_rate=float(healthy.accepted_frozen.mean()) if len(healthy) else np.nan,
                         frozen_operating_point=True))
    return pd.DataFrame(rows)


def stress_summary(records):
    rows = []
    columns = ["variant", "regime", "N", "mode", "mechanism"]
    subset = records[(records.task == "stress") & (records.mechanism != "healthy")]
    for keys, frame in subset.groupby(columns):
        rows.append(dict(zip(columns, keys), n_batches=len(frame),
                         n_detected=int(frame.detected.sum()),
                         n_five_class_reports=int(frame.accepted_frozen.sum()),
                         detection=float(frame.detected.mean()),
                         withheld_fraction=float(1 - frame.accepted_frozen.mean()),
                         five_class_reporting_rate=float(frame.accepted_frozen.mean())))
    return pd.DataFrame(rows)


def save_figures(curve, severity, output, severity_sensor=None):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    path = output / "figures"
    path.mkdir(exist_ok=True)
    for regime in curve.regime.unique() if len(curve) else []:
        subset = curve[(curve.regime == regime) & (curve.task == "baseline")]
        if not len(subset):
            continue
        fig, ax = plt.subplots(figsize=(6.2, 4.2))
        method_names = {"candidate": "Candidate replacement", "adjusted": "Largest adjusted error",
                        "covariance": "Covariance-adjusted error", "pca": "PCA reconstruction",
                        "autoencoder": "Autoencoder reconstruction"}
        for variant, part in subset.groupby("variant"):
            trace = part[~part.frozen_operating_point].sort_values("coverage")
            ax.plot(trace.coverage, trace.precision, label=method_names.get(variant, variant))
            frozen = part[part.frozen_operating_point]
            ax.scatter(frozen.coverage, frozen.precision, s=35)
        ax.set(xlabel="Fraction of groups receiving a diagnosis", ylabel="Correctness among reported diagnoses",
               xlim=(0, 1), ylim=(0, 1.01))
        ax.grid(alpha=0.2)
        ax.legend(frameon=False)
        fig.tight_layout()
        for extension in ("png", "pdf"):
            fig.savefig(path / f"baseline_reporting_{regime}.{extension}", dpi=180)
        plt.close(fig)
    for regime in severity.regime.unique() if len(severity) else []:
        selected = severity[(severity.regime == regime) & (severity.view == "all_injections")]
        mechanisms = [m for m in ("bias", "gain", "saturation") if m in selected.mechanism.unique()]
        if not mechanisms:
            continue
        fig, axes = plt.subplots(2, len(mechanisms), figsize=(4.0 * len(mechanisms), 6.1), squeeze=False, sharex="col")
        if severity_sensor:
            names = {"T_IM": "Intake-manifold temperature", "P_IM": "Intake-manifold pressure",
                     "EGR_Rate": "Exhaust-gas recirculation rate", "MF_IA": "Intake-air mass flow",
                     "NOx_EO": "Engine-out NOx concentration"}
            fig.suptitle(names.get(severity_sensor, severity_sensor))
        for j, mechanism in enumerate(mechanisms):
            part = selected[selected.mechanism == mechanism]
            for direction, d in part.groupby("direction"):
                d = d.sort_values("severity")
                label = "positive" if direction == 1 else "negative"
                suffix = f" ({label})" if mechanism in ("bias", "gain") else ""
                axes[0, j].plot(d.severity, d.detection, marker="o", label=f"Detection{suffix}")
                axes[0, j].plot(d.severity, d.reporting_rate, marker="s", linestyle="--", label=f"Diagnoses issued{suffix}")
                if d.complete_correct_given_report.notna().any():
                    axes[1, j].plot(d.severity, d.complete_correct_given_report, marker="o", label=label if suffix else "Clipping")
            if not part.complete_correct_given_report.notna().any():
                axes[1, j].text(0.5, 0.5, "No diagnoses issued", transform=axes[1, j].transAxes,
                                ha="center", va="center", fontsize=10)
            titles = {"bias": "Offset error", "gain": "Scaling error", "saturation": "Clipping"}
            axes[0, j].set_title(titles[mechanism])
            labels = {"saturation": "Training percentile for cap",
                      "bias": "Offset / training standard deviation",
                      "gain": "Fractional sensor gain change"}
            axes[1, j].set_xlabel(labels[mechanism])
            axes[0, j].set_ylabel("Fraction of groups")
            axes[1, j].set_ylabel("Correctness among reports")
            for ax in axes[:, j]:
                ax.set_ylim(0, 1.01)
                ax.grid(alpha=0.2)
                if ax.get_legend_handles_labels()[0]:
                    ax.legend(frameon=False, fontsize=8)
        fig.tight_layout()
        for extension in ("png", "pdf"):
            sensor_suffix = f"_{severity_sensor}" if severity_sensor else ""
            fig.savefig(path / f"final_pipeline_severity{sensor_suffix}_{regime}.{extension}", dpi=180)
        plt.close(fig)


def prepare_models(train, cfg, regime, fold, variants, quick, cache):
    final_path = cache / "final_models.joblib"
    if final_path.exists():
        final = joblib.load(final_path)
    else:
        log(f"seed {cfg.seed} {regime} fold {fold}: fitting healthy-map models")
        final = SourceModels(ep.derive_seed(cfg.seed, fold, "final"), variants, quick).fit(train)
        final.training_medians = train[MONITORED].median().to_dict()
        final.training_context_mean = train[CONTEXT].mean().values
        final.training_context_std = np.maximum(train[CONTEXT].std().values, 1e-9)
        atomic_joblib(final, final_path)
    prepared = []
    for i, (fit_indices, validation_indices) in enumerate(ep.inner_splits(train, regime, cfg)):
        path = cache / f"inner_{i}.joblib"
        if path.exists():
            prepared.append(joblib.load(path))
            continue
        fit = train.iloc[fit_indices]
        validation = train.iloc[validation_indices]
        log(f"seed {cfg.seed} {regime} fold {fold}: development partition {i + 1}/{cfg.n_inner}")
        models = SourceModels(ep.derive_seed(cfg.seed, fold, i), variants, quick).fit(fit)
        grid = ep.DEV_GRID
        if quick:
            grid = {m: [values[len(values) // 2]] for m, values in grid.items()}
        scenarios = score_scenarios(models, validation, grid, cfg, regime, fold, "inner")
        disjoint = not set(fit.index).intersection(validation.index)
        if not disjoint:
            raise AssertionError("Healthy-map model training overlaps its scored development observations")
        item = dict(inner_fold=i, models=models, scenarios=scenarios,
                    row_ids=validation.index.values, audit_disjoint=disjoint)
        atomic_joblib(item, path)
        prepared.append(item)
    return final, prepared


def task_settings(args):
    settings = []
    if "baseline" in args.tasks:
        settings.append(("baseline", 20, "diverse", args.variants))
    if "severity" in args.tasks:
        settings.append(("severity", 20, "diverse", ["candidate"]))
    if "diversity" in args.tasks:
        for size in args.batch_sizes:
            for mode in ("diverse", "local"):
                settings.append(("diversity", size, mode, ["candidate"]))
    if "stress" in args.tasks:
        settings.append(("stress", 20, "diverse", ["candidate"]))
    return settings


def run(args):
    output = Path(args.output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    frame, _ = load_engine_map(args.data_path)
    frame = assign_speed_blocks(frame)
    frame["rpm_level"] = frame["Engine Speed"].round(0)
    all_variants = sorted(set(args.variants) | {"candidate"})
    source_fingerprint = hashlib.sha256()
    for source in [Path(__file__), Path(ep.__file__),
                   Path(ep.__file__).parents[1] / "health" / "shi.py",
                   Path(ep.__file__).parents[1] / "health" / "witness_isolator.py"]:
        source_fingerprint.update(source.read_bytes())
    config_args = vars(args).copy()
    for name in ("data_path", "output"):
        config_args[name] = _path_label(config_args[name])
    config = dict(version=VERSION, args=config_args,
                  data_sha256=hashlib.sha256(Path(args.data_path).read_bytes()).hexdigest(),
                  source_sha256=source_fingerprint.hexdigest())
    digest = hashlib.sha256(json.dumps(config, sort_keys=True).encode()).hexdigest()[:12]
    cache_root = output / "cache" / digest
    run_records = output / "records" / digest
    cache_root.mkdir(parents=True, exist_ok=True)
    run_records.mkdir(parents=True, exist_ok=True)
    configuration_file = output / "configuration.json"
    previous_configuration = json.loads(configuration_file.read_text()) if configuration_file.exists() else None
    configuration_file.write_text(json.dumps(config, indent=2), encoding="utf-8")
    if args.quick:
        log("Smoke run: limited folds and generation")
    chunks, audits, calibrations, routes = [], [], [], []
    calibration_file = output / "calibration.csv"
    if previous_configuration == config and calibration_file.exists() and calibration_file.stat().st_size > 1:
        calibrations = pd.read_csv(calibration_file, float_precision="round_trip").to_dict("records")
    settings = task_settings(args)
    for seed in args.seeds:
        cfg = ep.EpisodeConfig(seed=seed, episode_sizes=tuple(args.batch_sizes),
                               modes=("diverse", "local"), n_inner=3 if args.quick else 4,
                               n_realizations=args.realizations, n_episodes=args.episodes,
                               n_healthy_episodes=args.healthy_episodes,
                               shi_normalization="ensemble", min_accepted=10 if args.quick else 20)
        regimes = {}
        if "heldout" in args.regimes:
            regimes[REGIMES["heldout"]] = list(GroupKFold(min(7, len(frame.rpm_level.unique()))).split(frame, groups=frame.rpm_level))
        if "within" in args.regimes:
            regimes[REGIMES["within"]] = list(KFold(5, shuffle=True, random_state=seed).split(frame))
        for regime, splits in regimes.items():
            for fold, (training_indices, test_indices) in enumerate(splits):
                if args.fold_limit and fold >= args.fold_limit:
                    break
                train, test = frame.iloc[training_indices].copy(), frame.iloc[test_indices].copy()
                if set(train.index).intersection(test.index):
                    raise AssertionError("Outer fold observation overlap")
                if regime == REGIMES["heldout"] and set(train.rpm_level).intersection(test.rpm_level):
                    raise AssertionError("Held-out engine-speed levels overlap training speeds")
                fold_cache = cache_root / f"seed{seed}_{regime}_fold{fold}"
                fold_cache.mkdir(parents=True, exist_ok=True)
                models, prepared = prepare_models(train, cfg, regime, fold, all_variants, args.quick, fold_cache)
                for candidate, route in models.base.iso.routes.items():
                    routes.append(dict(seed=seed, regime=regime, outer_fold=fold,
                                       candidate_sensor=candidate,
                                       reconstruction_inputs=";".join(CONTEXT + sorted(route["S"])),
                                       checking_target=route["wt"],
                                       checking_inputs=";".join(CONTEXT + list(route["wi"])),
                                       training_selection_score=float(route["cv"])))
                audits.append(dict(seed=seed, regime=regime, outer_fold=fold,
                                   n_train=len(train), n_test=len(test),
                                   train_speeds=";".join(map(str, sorted(train.rpm_level.unique()))),
                                   test_speeds=";".join(map(str, sorted(test.rpm_level.unique()))),
                                   outer_observations_disjoint=True,
                                   inner_observations_disjoint=all(pr["audit_disjoint"] for pr in prepared),
                                   saturation_caps_training_only=True,
                                   autoencoder_converged=getattr(models, "autoencoder_converged", np.nan)))
                scenario_cache = {}
                for task, N, mode, variants in settings:
                    for variant in variants:
                        chunk_path = run_records / f"seed{seed}_{regime}_fold{fold}_{task}_{N}_{mode}_{variant}.csv"
                        if chunk_path.exists():
                            log(f"resume {chunk_path.name}")
                            chunks.append(pd.read_csv(chunk_path, float_precision="round_trip"))
                            continue
                        if N > len(test):
                            log(f"skip N={N}: only {len(test)} distinct held-out points")
                            continue
                        report_path = fold_cache / f"reporting_{N}_{mode}_{variant}.joblib"
                        if report_path.exists():
                            deployment = joblib.load(report_path)
                        else:
                            inner_records = []
                            for partition in prepared:
                                if N > len(partition["row_ids"]):
                                    raise ValueError(f"N={N} exceeds development partition size {len(partition['row_ids'])}")
                                current = episode_records(partition["models"], partition["scenarios"], N, mode,
                                                          cfg, regime, fold, "inner", partition["row_ids"], variant)
                                for record in current:
                                    record["inner_fold"] = partition["inner_fold"]
                                inner_records.extend(current)
                            healthy = np.asarray([r["n_suspicious"] for r in inner_records if r["mech"] == "healthy"])
                            alarm = ep.calibrate_k_alarm(healthy, N, cfg)
                            log(f"seed {seed} {regime} fold {fold}: fit {variant}, {N} points, {mode}")
                            deployment = fit_reporting(inner_records, alarm["k_alarm"], cfg,
                                                        trees=40 if args.quick else 300, threads=args.threads,
                                                        population=args.reporting_population)
                            deployment["alarm"] = alarm
                            atomic_joblib(deployment, report_path)
                        calibrations.append(dict(seed=seed, regime=regime, outer_fold=fold,
                                                 variant=variant, N=N, mode=mode,
                                                 k_alarm=deployment["alarm"]["k_alarm"],
                                                 alarm_status=deployment["alarm"]["status"],
                                                 reporting_threshold=deployment["threshold"]["thr"],
                                                 reporting_status=deployment["threshold"]["status"],
                                                 n_development=deployment["n_development"],
                                                 n_development_faults=deployment.get("n_development_faults", np.nan),
                                                 n_development_healthy=deployment.get("n_development_healthy", 0),
                                                 n_development_nonmanifested=deployment.get("n_development_nonmanifested", 0),
                                                 reporting_population=deployment.get("reporting_population", "fault_only"),
                                                 development_precision=deployment["threshold"].get("prec", np.nan),
                                                 development_precision_lower=deployment["threshold"].get("prec_lo", np.nan)))
                        scenario_key = "severity" if task == "severity" else "stress" if task == "stress" else "test"
                        if scenario_key not in scenario_cache:
                            if task == "stress":
                                scenarios = score_stress_scenarios(models, test, cfg, regime, fold)
                            else:
                                grid = SEVERITY_GRID if task == "severity" else ep.TEST_GRID
                                if args.quick:
                                    grid = ({"bias": [0.5, 1.5], "gain": [0.05, 0.16], "noise": [],
                                             "saturation": [80, 92], "partial_dropout": []}
                                            if task == "severity" else
                                            {m: [values[0]] for m, values in grid.items()})
                                log(f"seed {seed} {regime} fold {fold}: score {scenario_key} fault scenarios")
                                scenarios = score_scenarios(models, test, grid, cfg, regime, fold, "outer")
                            scenario_cache[scenario_key] = scenarios
                        records = episode_records(models, scenario_cache[scenario_key], N, mode, cfg,
                                                  regime, fold, "outer", test.index.values, variant)
                        predicted = predict_records(records, deployment, deployment["alarm"]["k_alarm"],
                                                    seed, variant, task)
                        atomic_csv(predicted, chunk_path)
                        chunks.append(predicted)
                        log(f"saved {len(predicted)} batch predictions: {task}, {variant}, N={N}, {mode}")

                atomic_csv(pd.DataFrame(audits), output / "fold_audit.csv")
                atomic_csv(pd.DataFrame(calibrations).drop_duplicates(), output / "calibration.csv")
                atomic_csv(pd.DataFrame(routes), output / "selected_sensor_routes.csv")
                if chunks:
                    atomic_csv(pd.concat(chunks, ignore_index=True), output / "batch_predictions.csv")
    if not chunks:
        raise RuntimeError("No test batches were produced; inspect fold and batch-size settings")
    predictions = pd.concat(chunks, ignore_index=True)
    summary = summaries(predictions)
    atomic_csv(summary, output / "summary_exact_counts.csv")

    fold_rows = []
    for (seed, fold), part in predictions.groupby(["seed", "outer_fold"]):
        table = summaries(part)
        table["seed"] = seed
        table["outer_fold"] = fold
        fold_rows.append(table)
    atomic_csv(pd.concat(fold_rows, ignore_index=True), output / "fold_summary_exact_counts.csv")
    atomic_csv(healthy_summary(predictions), output / "healthy_reporting.csv")
    curve = risk_coverage(predictions)
    atomic_csv(curve, output / "risk_coverage.csv")
    severity = severity_summary(predictions)
    if len(severity):
        atomic_csv(severity, output / "fault_severity.csv")
    sensor_severity = severity_summary(predictions, per_sensor=True)
    if len(sensor_severity):
        atomic_csv(sensor_severity, output / "fault_severity_by_sensor.csv")
    stress = stress_summary(predictions)
    if len(stress):
        atomic_csv(stress, output / "unfamiliar_faults.csv")
    save_figures(curve, severity, output)
    if len(sensor_severity):
        for sensor, selected in sensor_severity.groupby("true_src"):
            save_figures(pd.DataFrame(), selected, output, severity_sensor=sensor)
    log(f"Finished. Generated artifacts are in {output}")
    return 0


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data-path", default="data/engine_map.csv")
    p.add_argument("--output", default="outputs/primary")
    p.add_argument("--quick", action="store_true", help="one outer fold, short grids and small fitting models")
    p.add_argument("--seeds", nargs="+", type=int, default=[42])
    p.add_argument("--fold-limit", type=int, default=0, help="0 runs every outer fold")
    p.add_argument("--tasks", nargs="+", choices=["baseline", "severity", "diversity", "stress"],
                   default=["baseline", "severity", "diversity", "stress"])
    p.add_argument("--variants", nargs="+", choices=VARIANTS, default=list(VARIANTS),
                   help="matched source-attribution variants; every variant refits downstream models")
    p.add_argument("--regimes", nargs="+", choices=list(REGIMES), default=["heldout"])
    p.add_argument("--batch-sizes", nargs="+", type=int, default=[5, 10, 20])
    p.add_argument("--episodes", type=int, default=None, help="fault batches per scenario and configuration")
    p.add_argument("--healthy-episodes", type=int, default=None)
    p.add_argument("--realizations", type=int, default=None)
    p.add_argument("--threads", type=int, default=1, help="threads per random forest")
    p.add_argument("--reporting-population", choices=["fault_only", "fault_and_healthy"],
                   default="fault_and_healthy", help="calibration examples; typing always trains only on faults")
    return p


def main(argv=None):
    args = parser().parse_args(argv)
    args.seeds = list(dict.fromkeys(args.seeds))
    args.tasks = list(dict.fromkeys(args.tasks))
    args.variants = list(dict.fromkeys(args.variants))
    args.batch_sizes = list(dict.fromkeys(args.batch_sizes))
    args.regimes = list(dict.fromkeys(args.regimes))
    if args.quick and not args.fold_limit:
        args.fold_limit = 1
    args.episodes = args.episodes if args.episodes is not None else (2 if args.quick else 15)
    args.healthy_episodes = args.healthy_episodes if args.healthy_episodes is not None else (40 if args.quick else 120)
    args.realizations = args.realizations if args.realizations is not None else (1 if args.quick else 2)
    if args.quick and args.batch_sizes == [5, 10, 20]:
        args.batch_sizes = [5, 20]
    if min(args.batch_sizes + [args.episodes, args.healthy_episodes, args.realizations]) < 1:
        raise ValueError("Batch sizes and generation counts must be positive")
    if args.fold_limit < 0 or args.threads < 1 or any(seed < 0 or seed >= 2 ** 32 for seed in args.seeds):
        raise ValueError("Fold limit must be nonnegative, threads positive, and seeds in [0, 2**32)")
    return run(args)


if __name__ == "__main__":
    raise SystemExit(main())

