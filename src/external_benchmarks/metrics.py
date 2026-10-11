import numpy as np
import pandas as pd


def wilson_lower(k, n, z=1.96):

    if n == 0:
        return 0.0
    p = k / n
    d = 1 + z * z / n
    centre = p + z * z / (2 * n)
    margin = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return max(0.0, (centre - margin) / d)


def detection_metrics(scores, labels, threshold=None):

    from sklearn.metrics import roc_auc_score, average_precision_score
    scores = np.asarray(scores, float)
    labels = np.asarray(labels, int)
    out = {}
    if len(np.unique(labels)) == 2:
        out['auroc'] = float(roc_auc_score(labels, scores))
        out['auprc'] = float(average_precision_score(labels, scores))
    else:
        out['auroc'] = float('nan')
        out['auprc'] = float('nan')
    if threshold is not None:
        alarm = scores >= threshold
        pos, neg = labels == 1, labels == 0
        out['recall'] = float(alarm[pos].mean()) if pos.any() else float('nan')
        out['healthy_false_alarm'] = float(alarm[neg].mean()) if neg.any() else float('nan')
    return out


def detection_delay(alarm_flags, onset_index, sample_period_s=1.0):

    alarm = np.asarray(alarm_flags, bool)
    after = np.flatnonzero(alarm[onset_index:])
    if len(after) == 0:
        return dict(delay_samples=float('nan'), delay_seconds=float('nan'), detected=False)
    d = int(after[0])
    return dict(delay_samples=d, delay_seconds=d * sample_period_s, detected=True)


def source_metrics(predicted, truth, detected):

    predicted = np.asarray(predicted, dtype=object)
    truth = np.asarray(truth, dtype=object)
    detected = np.asarray(detected, bool)
    correct = predicted == truth
    return dict(
        source_given_detected=float(correct[detected].mean()) if detected.any() else float('nan'),
        end_to_end_source=float((correct & detected).mean()) if len(correct) else float('nan'),
        n_detected=int(detected.sum()), n_total=int(len(correct)))


def source_stability(predictions_over_time):

    preds = [p for p in predictions_over_time if p is not None]
    if not preds:
        return dict(stability=float('nan'), entropy=float('nan'))
    values, counts = np.unique(np.array(preds, dtype=object), return_counts=True)
    frac = counts / counts.sum()
    return dict(stability=float(frac.max()),
                entropy=float(-(frac * np.log(frac)).sum()))


COUNTERFACTUAL_FIELDS = [
    'shi_reduction',
    'joint_loglik_gain',
    'channels_resolved',
    'residual_anomaly_remaining',
    'downstream_error_reduction',
]


def counterfactual_evidence(before, after, n_anomalous_before, n_anomalous_after,
                            downstream_before=None, downstream_after=None):


    ev = {
        'shi_reduction': float(before['shi'] - after['shi']),
        'joint_loglik_gain': float(after['joint_loglik'] - before['joint_loglik']),
        'channels_resolved': int(n_anomalous_before - n_anomalous_after),
        'residual_anomaly_remaining': float(after['shi']),
    }
    if downstream_before is not None and downstream_after is not None:
        ev['downstream_error_reduction'] = float(downstream_before - downstream_after)
    else:
        ev['downstream_error_reduction'] = float('nan')
    return ev


def restoration_score(evidence, weights=None):


    w = weights or dict(shi_reduction=1.0, joint_loglik_gain=0.1,
                        channels_resolved=0.5, residual_anomaly_remaining=-1.0)
    return float(sum(w.get(k, 0.0) * evidence.get(k, 0.0)
                     for k in w if np.isfinite(evidence.get(k, np.nan))))


def sensor_vs_system_metrics(scores, is_sensor_fault, threshold=None):


    scores = np.asarray(scores, float)
    is_sensor = np.asarray(is_sensor_fault, int)
    out = detection_metrics(scores, is_sensor, threshold)
    if threshold is not None:
        called_sensor = scores >= threshold
        system = is_sensor == 0
        sensor = is_sensor == 1
        out['sensor_fault_recall'] = float(called_sensor[sensor].mean()) if sensor.any() else float('nan')
        out['system_fault_recall'] = float((~called_sensor)[system].mean()) if system.any() else float('nan')
        out['false_sensor_replacement_rate'] = (
            float(called_sensor[system].mean()) if system.any() else float('nan'))
    return out


def repair_metrics(error_before, error_after, error_oracle=None, repaired=None):


    before = np.asarray(error_before, float)
    after = np.asarray(error_after, float)
    mask = np.ones(len(before), bool) if repaired is None else np.asarray(repaired, bool)
    if not mask.any():
        return dict(repair_coverage=0.0, mean_error_before=float('nan'),
                    mean_error_after=float('nan'), excess_error_recovered=float('nan'),
                    actual_harm_rate=float('nan'))
    b, a = before[mask], after[mask]
    out = dict(
        repair_coverage=float(mask.mean()),
        mean_error_before=float(b.mean()),
        mean_error_after=float(a.mean()),
        actual_harm_rate=float((a > b).mean()),
    )
    if error_oracle is not None:
        o = np.asarray(error_oracle, float)[mask]
        denom = (b - o)
        good = np.abs(denom) > 1e-12
        out['excess_error_recovered'] = (
            float(np.mean((b[good] - a[good]) / denom[good])) if good.any() else float('nan'))
    else:
        out['excess_error_recovered'] = float('nan')
    return out


def selective_metrics(scores, correct, threshold):

    scores = np.asarray(scores, float)
    correct = np.asarray(correct, int)
    accepted = scores >= threshold
    n = int(accepted.sum())
    if n == 0:
        return dict(coverage=0.0, accepted_precision=float('nan'),
                    wilson_lower=float('nan'), n_accepted=0)
    k = int(correct[accepted].sum())
    return dict(coverage=float(accepted.mean()), accepted_precision=k / n,
                wilson_lower=float(wilson_lower(k, n)), n_accepted=n)


NA = 'NA'


def na_row(dataset, **kw):

    base = dict(dataset=dataset, detection=NA, source_accuracy=NA,
                sensor_vs_system=NA, selective_coverage=NA,
                selective_precision=NA, repair_harm=NA)
    base.update(kw)
    return base
