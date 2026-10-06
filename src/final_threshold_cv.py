"""
final_threshold_cv.py

1. Chooses the ensemble decision threshold using TRAINING data only (out-of-fold
   predictions), then evaluates that single threshold on the test set once.
2. Bootstrap 95% intervals on the test set for ROC-AUC and PR-AUC of all five models,
   plus paired bootstrap differences against the stacking ensemble.
3. Repeated stratified 5-fold cross-validation on the training set for all five models,
   with the Nadeau-Bengio corrected resampled t-test against the stacking ensemble.

Run from the project root:
    python src/final_threshold_cv.py

Expected time: 15 to 40 minutes (the ensemble is refitted in every fold).
Set QUICK = True below for a faster check (3 folds x 1 repeat is NOT enough for the paper).
"""
import os
import sys
import warnings

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.base import clone
from sklearn.metrics import (average_precision_score, brier_score_loss, matthews_corrcoef,
                             roc_auc_score)
from sklearn.model_selection import (RepeatedStratifiedKFold, StratifiedKFold, cross_validate,
                                     cross_val_predict, train_test_split)

warnings.filterwarnings("ignore")
sys.path.insert(0, os.getcwd())
from explainability import load_default_pipeline  # noqa: E402

SEED = 42
RESULTS = "results"
N_BOOT = 2000
N_SPLITS = 5
N_REPEATS = 3
QUICK = False
if QUICK:
    N_REPEATS = 1
os.makedirs(RESULTS, exist_ok=True)
lines = []


def say(t=""):
    print(t)
    lines.append(t)


# ---------------------------------------------------------------- data
data = pd.read_csv(os.path.join("dataset", "healthcare-dataset-stroke-data.csv")).drop(columns=["id"])
X = data.drop(columns=["stroke"])
y = data["stroke"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=SEED, stratify=y)
yt = y_test.values
ytr = y_train.values

MODELS = {
    "LR": "logistic_regression",
    "RF": "random_forest",
    "XGB": "xgboost",
    "Tuned XGB": "tuned_xgboost",
    "Stacking": "stacking_ensemble",
}
fitted = {k: load_default_pipeline(v) for k, v in MODELS.items()}
ens = fitted["Stacking"]


def metrics_at(y_true, p, t):
    pred = (p >= t).astype(int)
    tp = int(((pred == 1) & (y_true == 1)).sum())
    fp = int(((pred == 1) & (y_true == 0)).sum())
    fn = int(((pred == 0) & (y_true == 1)).sum())
    tn = int(((pred == 0) & (y_true == 0)).sum())
    sens = tp / max(tp + fn, 1)
    spec = tn / max(tn + fp, 1)
    prec = tp / max(tp + fp, 1)
    f1 = 2 * prec * sens / (prec + sens) if (prec + sens) > 0 else 0.0
    mcc = matthews_corrcoef(y_true, pred) if (pred.sum() > 0) else 0.0
    return dict(threshold=t, sensitivity=sens, specificity=spec, precision=prec, f1=f1, mcc=mcc,
                tp=tp, fp=fp, fn=fn, tn=tn)


# ---------------------------------------------------------------- 1. threshold from training data only
say("Step 1: out-of-fold probabilities for the stacking ensemble on the TRAINING set ...")
oof = cross_val_predict(
    clone(ens), X_train, y_train,
    cv=StratifiedKFold(5, shuffle=True, random_state=SEED), method="predict_proba", n_jobs=1,
)[:, 1]
np.save(os.path.join(RESULTS, "final_oof_ensemble_probs.npy"), oof)

grid = np.round(np.arange(0.02, 0.40, 0.005), 3)
oof_tab = pd.DataFrame([metrics_at(ytr, oof, t) for t in grid])
oof_tab.to_csv(os.path.join(RESULTS, "final_oof_threshold_table.csv"), index=False)

t_mcc = float(oof_tab.loc[oof_tab.mcc.idxmax(), "threshold"])
ok70 = oof_tab[oof_tab.sensitivity >= 0.70]
t_s70 = float(ok70.threshold.max()) if len(ok70) else float(grid[0])
say("OOF-selected thresholds: best MCC -> %.3f ; largest threshold with OOF sensitivity >= 0.70 -> %.3f"
    % (t_mcc, t_s70))

p_ens = ens.predict_proba(X_test)[:, 1]
rng = np.random.RandomState(SEED)
pos = np.where(yt == 1)[0]
neg = np.where(yt == 0)[0]
boots = [np.concatenate([rng.choice(pos, len(pos)), rng.choice(neg, len(neg))]) for _ in range(N_BOOT)]

rows = []
for label, t in [("best MCC on training OOF", t_mcc), ("sensitivity >= 0.70 on training OOF", t_s70)]:
    m = metrics_at(yt, p_ens, t)
    bs = pd.DataFrame([metrics_at(yt[b], p_ens[b], t) for b in boots])
    row = {"rule": label, **m}
    for c in ["sensitivity", "specificity", "precision", "f1", "mcc"]:
        row[c + "_lo"], row[c + "_hi"] = np.percentile(bs[c], [2.5, 97.5])
    rows.append(row)
    say("  Test set at %.3f (%s): sens %.3f [%.3f, %.3f], spec %.3f [%.3f, %.3f], precision %.3f, MCC %.3f, TP %d FP %d"
        % (t, label, m["sensitivity"], row["sensitivity_lo"], row["sensitivity_hi"], m["specificity"],
           row["specificity_lo"], row["specificity_hi"], m["precision"], m["mcc"], m["tp"], m["fp"]))
pd.DataFrame(rows).to_csv(os.path.join(RESULTS, "final_threshold_selection.csv"), index=False)

# ---------------------------------------------------------------- 2. test-set intervals and paired differences
def ece(y_true, p, bins=10):
    e, n = 0.0, len(y_true)
    for i in range(bins):
        lo, hi = i / bins, (i + 1) / bins
        m = (p >= lo) & ((p < hi) if i < bins - 1 else (p <= hi))
        if m.sum():
            e += m.sum() / n * abs(y_true[m].mean() - p[m].mean())
    return e


probs = {k: v.predict_proba(X_test)[:, 1] for k, v in fitted.items()}
say("\nStep 2: test-set discrimination with 95%% bootstrap intervals (%d resamples)" % N_BOOT)
ci_rows = []
auc_b = {k: np.array([roc_auc_score(yt[b], p[b]) for b in boots]) for k, p in probs.items()}
ap_b = {k: np.array([average_precision_score(yt[b], p[b]) for b in boots]) for k, p in probs.items()}
for k, p in probs.items():
    r = {"model": k, "roc_auc": roc_auc_score(yt, p), "roc_auc_lo": np.percentile(auc_b[k], 2.5),
         "roc_auc_hi": np.percentile(auc_b[k], 97.5), "pr_auc": average_precision_score(yt, p),
         "pr_auc_lo": np.percentile(ap_b[k], 2.5), "pr_auc_hi": np.percentile(ap_b[k], 97.5),
         "brier": brier_score_loss(yt, p), "ece10": ece(yt, p)}
    if k != "Stacking":
        d_auc = auc_b["Stacking"] - auc_b[k]
        d_ap = ap_b["Stacking"] - ap_b[k]
        r.update({"ens_minus_model_auc": roc_auc_score(yt, probs["Stacking"]) - r["roc_auc"],
                  "ens_minus_model_auc_lo": np.percentile(d_auc, 2.5),
                  "ens_minus_model_auc_hi": np.percentile(d_auc, 97.5),
                  "ens_minus_model_ap": average_precision_score(yt, probs["Stacking"]) - r["pr_auc"],
                  "ens_minus_model_ap_lo": np.percentile(d_ap, 2.5),
                  "ens_minus_model_ap_hi": np.percentile(d_ap, 97.5)})
    ci_rows.append(r)
    say("  %-10s ROC-AUC %.3f [%.3f, %.3f]  PR-AUC %.3f [%.3f, %.3f]  Brier %.4f  ECE %.4f"
        % (k, r["roc_auc"], r["roc_auc_lo"], r["roc_auc_hi"], r["pr_auc"], r["pr_auc_lo"], r["pr_auc_hi"],
           r["brier"], r["ece10"]))
ci_df = pd.DataFrame(ci_rows)
ci_df.to_csv(os.path.join(RESULTS, "final_test_ci.csv"), index=False)
say("Paired bootstrap, Stacking minus model (interval containing 0 = no clear difference):")
for r in ci_rows:
    if r["model"] == "Stacking":
        continue
    say("  vs %-10s dAUC %+.3f [%+.3f, %+.3f]   dPR-AUC %+.3f [%+.3f, %+.3f]"
        % (r["model"], r["ens_minus_model_auc"], r["ens_minus_model_auc_lo"], r["ens_minus_model_auc_hi"],
           r["ens_minus_model_ap"], r["ens_minus_model_ap_lo"], r["ens_minus_model_ap_hi"]))

# ---------------------------------------------------------------- 3. repeated CV on the training set
say("\nStep 3: repeated stratified %d-fold CV x %d repeats on the training set ..." % (N_SPLITS, N_REPEATS))
cv = RepeatedStratifiedKFold(n_splits=N_SPLITS, n_repeats=N_REPEATS, random_state=SEED)
fold_rows, scores = [], {}
for k, pipe in fitted.items():
    print("  cross-validating %s ..." % k)
    res = cross_validate(clone(pipe), X_train, y_train, cv=cv, scoring=["roc_auc", "average_precision"], n_jobs=1)
    scores[k] = res
    for i in range(len(res["test_roc_auc"])):
        fold_rows.append({"model": k, "fold": i, "roc_auc": res["test_roc_auc"][i],
                          "pr_auc": res["test_average_precision"][i]})
fold_df = pd.DataFrame(fold_rows)
fold_df.to_csv(os.path.join(RESULTS, "final_repeated_cv_folds.csv"), index=False)
summ = fold_df.groupby("model").agg(roc_auc_mean=("roc_auc", "mean"), roc_auc_sd=("roc_auc", "std"),
                                    pr_auc_mean=("pr_auc", "mean"), pr_auc_sd=("pr_auc", "std")).reset_index()
summ.to_csv(os.path.join(RESULTS, "final_repeated_cv_summary.csv"), index=False)
say("Cross-validated (mean +/- SD over %d folds):" % (N_SPLITS * N_REPEATS))
for r in summ.itertuples():
    say("  %-10s ROC-AUC %.3f +/- %.3f   PR-AUC %.3f +/- %.3f"
        % (r.model, r.roc_auc_mean, r.roc_auc_sd, r.pr_auc_mean, r.pr_auc_sd))

# Nadeau-Bengio corrected resampled t-test, Stacking versus each other model
k_folds = N_SPLITS * N_REPEATS
ratio = 1.0 / (N_SPLITS - 1)          # n_test / n_train
pt_rows = []
say("Corrected resampled t-test, Stacking minus model (Nadeau and Bengio):")
for k in MODELS:
    if k == "Stacking":
        continue
    for met, key in [("roc_auc", "test_roc_auc"), ("pr_auc", "test_average_precision")]:
        d = np.asarray(scores["Stacking"][key]) - np.asarray(scores[k][key])
        var = d.var(ddof=1)
        t_stat = d.mean() / np.sqrt((1.0 / k_folds + ratio) * var) if var > 0 else np.nan
        p = 2 * (1 - stats.t.cdf(abs(t_stat), k_folds - 1)) if np.isfinite(t_stat) else np.nan
        pt_rows.append({"model": k, "metric": met, "mean_diff": d.mean(), "t": t_stat, "p_value": p})
        say("  vs %-10s %-7s diff %+.4f  t %.2f  p %.3f" % (k, met, d.mean(), t_stat, p))
pd.DataFrame(pt_rows).to_csv(os.path.join(RESULTS, "final_paired_tests.csv"), index=False)
say("(8 comparisons were made; a Bonferroni-adjusted threshold is 0.05/8 = 0.00625)")

with open(os.path.join(RESULTS, "final_threshold_cv_summary.txt"), "w") as fh:
    fh.write("\n".join(lines))
print("\nDone. Send me the files starting with 'final_' from the results folder.")
